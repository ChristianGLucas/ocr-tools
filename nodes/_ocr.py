"""Shared OCR helpers for christiangeorgelucas/ocr-tools.

One place that knows how to (a) resolve the canonical `Image` envelope into a
decoded RGB pixel array — enforcing hard size bounds and an SSRF-guarded URL
fetch on untrusted input — and (b) drive the vendored, Apache-2.0 RapidOCR /
PP-OCRv4 ONNX engine and shape its raw output into plain Python structures the
nodes turn into protobuf messages.

The engine (three bundled ONNX models) is expensive to construct, so it is built
once and cached. It is read-only and holds no per-call state, so every node
invocation remains a pure function of its input.
"""
import io
import ipaddress
import os
import socket
import sys
import urllib.request
from urllib.parse import urlparse

import numpy as np
from PIL import Image as PILImage

# The vendored RapidOCR package lives under <package>/vendor so we can pin the
# exact code + models and drop RapidOCR's own non-permissive transitive dep
# (tqdm, MPL-2.0) which it never actually imports. Put it on the path.
_VENDOR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor"
)
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from rapidocr_onnxruntime import RapidOCR  # noqa: E402 (must follow sys.path insert)

# --- Hard bounds on untrusted input (fire on the RAW input before we allocate) --
# Max encoded image bytes accepted (inline or fetched) before any decode.
_MAX_BYTES = 20 * 1024 * 1024
# Max decoded pixel count (width * height). Caps decompression-bomb blow-up:
# checked against the header-declared dimensions BEFORE the pixels are decoded.
_MAX_PIXELS = 40_000_000
# Timeout for a URL fetch, in seconds.
_FETCH_TIMEOUT = 20

# Also clamp Pillow's own decompression-bomb guard to our stricter cap.
PILImage.MAX_IMAGE_PIXELS = _MAX_PIXELS


class OcrError(Exception):
    """A structured, caller-facing failure. Nodes turn this into an `error`
    field on their output message rather than letting it propagate as a crash."""


_ENGINE = None


def _engine() -> RapidOCR:
    """Return the process-wide RapidOCR engine, building it once on first use."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RapidOCR()
    return _ENGINE


def _is_disallowed_ip(ip: str) -> bool:
    """True if `ip` is not a normal public address (loopback, private, link-local,
    reserved, multicast, or unspecified) — the SSRF block list."""
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _fetch(url: str) -> bytes:
    """Fetch an image from an http/https URL with an SSRF guard and a size cap.

    Rejects non-http(s) schemes and any host that resolves to a private,
    loopback, link-local, reserved, multicast, or unspecified address — so the
    caller cannot use the `url` field to reach cloud metadata endpoints or
    internal services. Reads at most `_MAX_BYTES`.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise OcrError("url must be an http or https URL")
    host = parsed.hostname
    if not host:
        raise OcrError("url has no host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise OcrError("could not resolve url host")
    for info in infos:
        ip = info[4][0]
        if _is_disallowed_ip(ip):
            raise OcrError("url host resolves to a non-public address")
    req = urllib.request.Request(url, headers={"User-Agent": "axiom-ocr-tools/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:  # noqa: S310
            raw = resp.read(_MAX_BYTES + 1)
    except OcrError:
        raise
    except Exception as exc:  # network / HTTP error -> structured failure
        raise OcrError(f"failed to fetch url: {type(exc).__name__}")
    if len(raw) > _MAX_BYTES:
        raise OcrError("fetched image exceeds 20 MB limit")
    return raw


def _raw_bytes(image) -> bytes:
    """Resolve the `Image` envelope to raw encoded bytes (inline `data`, else
    fetch `url`), enforcing the byte cap. Raises OcrError on empty/oversized."""
    raw = bytes(image.data)
    if raw:
        if len(raw) > _MAX_BYTES:
            raise OcrError("image exceeds 20 MB limit")
        return raw
    if image.url:
        return _fetch(image.url)
    raise OcrError("image has neither `data` nor `url`")


def load_rgb(image) -> np.ndarray:
    """Resolve and decode an `Image` envelope into an (H, W, 3) uint8 RGB array.

    Enforces the byte cap on the encoded input and the pixel cap on the
    header-declared dimensions BEFORE decoding, so a small 'bomb' image with
    huge declared dimensions is rejected without allocating its pixels.
    """
    raw = _raw_bytes(image)
    try:
        pil = PILImage.open(io.BytesIO(raw))
    except Exception:
        raise OcrError("input is not a decodable image")
    w, h = pil.size
    if w <= 0 or h <= 0:
        raise OcrError("image has zero dimension")
    if w * h > _MAX_PIXELS:
        raise OcrError("image exceeds 40 megapixel limit")
    try:
        pil = pil.convert("RGB")
        arr = np.asarray(pil, dtype=np.uint8)
    except Exception:
        raise OcrError("input is not a decodable image")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise OcrError("input is not a decodable image")
    return arr


def _quad_and_bbox(box):
    """Turn a RapidOCR detection box (4 [x, y] points) into a list of (x, y)
    float tuples plus its axis-aligned (x0, y0, x1, y1) bounds."""
    pts = [(float(p[0]), float(p[1])) for p in box]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return pts, min(xs), min(ys), max(xs), max(ys)


def recognize(image) -> dict:
    """Detect + recognize all text in the image.

    Returns a dict: text (all lines joined by '\\n'), lines (list of per-line
    dicts with text, confidence, x0/y0/x1/y1, quad), mean_confidence, line_count.
    """
    arr = load_rgb(image)
    result, _ = _engine()(arr)
    lines = []
    if result:
        for box, text, score in result:
            quad, x0, y0, x1, y1 = _quad_and_bbox(box)
            lines.append(
                {
                    "text": str(text),
                    "confidence": float(score),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "quad": quad,
                }
            )
    text = "\n".join(ln["text"] for ln in lines)
    mean_conf = sum(ln["confidence"] for ln in lines) / len(lines) if lines else 0.0
    return {
        "text": text,
        "lines": lines,
        "mean_confidence": mean_conf,
        "line_count": len(lines),
    }


def detect_regions(image) -> list:
    """Run detection only (no recognition). Returns a list of region dicts with
    x0/y0/x1/y1 and quad — a fast 'where is the text' pass."""
    arr = load_rgb(image)
    out = _engine()(arr, use_det=True, use_rec=False, use_cls=False)
    boxes = out[0] if isinstance(out, tuple) else out
    regions = []
    if boxes:
        for box in boxes:
            quad, x0, y0, x1, y1 = _quad_and_bbox(box)
            regions.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "quad": quad})
    return regions
