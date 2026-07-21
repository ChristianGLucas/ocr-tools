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
import http.client
import io
import ipaddress
import os
import socket
import ssl
import sys
from urllib.parse import urljoin, urlparse

import cv2
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
# Max redirect hops to follow (each re-validated).
_MAX_REDIRECTS = 5

# Also clamp Pillow's own decompression-bomb guard to our stricter cap.
PILImage.MAX_IMAGE_PIXELS = _MAX_PIXELS


class OcrError(Exception):
    """A structured, caller-facing failure. Nodes turn this into an `error`
    field on their output message rather than letting it propagate as a crash."""


# Build the engine EAGERLY at import (i.e. at pod boot), not lazily on the first
# request. Loading the three ONNX models takes a second or two; doing it during
# module import keeps it off the request path, so the first real invocation is
# already warm and does not risk the (short) invocation deadline on a cold start.
# Guarded so an unexpected load failure defers to first use rather than breaking
# import (which would also break `axiom validate`/tooling).
try:
    _ENGINE = RapidOCR()
except Exception:  # pragma: no cover - defensive; model load is exercised in tests
    _ENGINE = None


def _engine() -> RapidOCR:
    """Return the process-wide RapidOCR engine (built once, eagerly at import)."""
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


def _validate_url(url: str):
    """Validate a URL for fetching and resolve it to a safe, pinned IP.

    Rejects non-http(s) schemes and any host that resolves to a private,
    loopback, link-local, reserved, multicast, or unspecified address. Returns
    (scheme, host, port, ip) where `ip` is a validated address we then connect
    to DIRECTLY — so the address checked is the address contacted, closing the
    DNS-rebinding window. Called on the initial URL AND on every redirect hop.
    Raises OcrError on anything disallowed.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise OcrError("url must be an http or https URL")
    host = parsed.hostname
    if not host:
        raise OcrError("url has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise OcrError("could not resolve url host")
    pinned = None
    for info in infos:
        cand = info[4][0]
        if _is_disallowed_ip(cand):
            raise OcrError("url host resolves to a non-public address")
        if pinned is None:
            pinned = cand
    if pinned is None:
        raise OcrError("could not resolve url host")
    return parsed.scheme, host, port, pinned


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that dials a pre-validated IP instead of re-resolving the
    host (so the address we vetted is the address we actually contact)."""

    def __init__(self, host, pinned_ip, **kwargs):
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that dials a pre-validated IP while still using the
    original hostname for TLS SNI and certificate verification."""

    def __init__(self, host, pinned_ip, *, context, **kwargs):
        super().__init__(host, context=context, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            sock = self.sock
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _fetch(url: str) -> bytes:
    """Fetch an image over http/https with an SSRF guard enforced on the initial
    URL AND re-enforced on every redirect hop, connecting only to the validated,
    pinned address. Follows at most `_MAX_REDIRECTS` redirects and reads at most
    `_MAX_BYTES` — so a public URL that 3xx-redirects to an internal/metadata
    address is blocked at the hop, not followed.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        scheme, host, port, ip = _validate_url(current)
        if scheme == "https":
            conn = _PinnedHTTPSConnection(
                host, ip, port=port, timeout=_FETCH_TIMEOUT,
                context=ssl.create_default_context(),
            )
        else:
            conn = _PinnedHTTPConnection(host, ip, port=port, timeout=_FETCH_TIMEOUT)
        try:
            parsed = urlparse(current)
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn.request("GET", path, headers={"User-Agent": "axiom-ocr-tools/0.1"})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.getheader("Location")
                resp.read(_MAX_BYTES + 1)  # drain, bounded
                if not loc:
                    raise OcrError("redirect without a Location header")
                current = urljoin(current, loc)
                continue
            if resp.status != 200:
                raise OcrError(f"fetch failed with HTTP {resp.status}")
            raw = resp.read(_MAX_BYTES + 1)
        except OcrError:
            raise
        except Exception as exc:  # network / TLS / HTTP error -> structured failure
            raise OcrError(f"failed to fetch url: {type(exc).__name__}")
        finally:
            conn.close()
        if len(raw) > _MAX_BYTES:
            raise OcrError("fetched image exceeds 20 MB limit")
        return raw
    raise OcrError("too many redirects")


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


# Formats the `Image` envelope (and its field-flattened siblings below) may
# declare/return, matching image-tools' documented convention exactly.
_VALID_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}


def _decode(raw: bytes):
    """Decode raw encoded image bytes into an (H, W, 3) uint8 RGB array plus
    the detected source format (one of `_VALID_FORMATS`, defaulting to "PNG"
    when undetected/unrecognized).

    Enforces the pixel cap on the header-declared dimensions BEFORE decoding
    pixels, so a small 'bomb' image with huge declared dimensions is rejected
    without allocating its pixel buffer.
    """
    try:
        pil = PILImage.open(io.BytesIO(raw))
    except Exception:
        raise OcrError("input is not a decodable image")
    w, h = pil.size
    if w <= 0 or h <= 0:
        raise OcrError("image has zero dimension")
    if w * h > _MAX_PIXELS:
        raise OcrError("image exceeds 40 megapixel limit")
    fmt = (pil.format or "PNG").upper()
    if fmt not in _VALID_FORMATS:
        fmt = "PNG"
    try:
        pil = pil.convert("RGB")
        arr = np.asarray(pil, dtype=np.uint8)
    except Exception:
        raise OcrError("input is not a decodable image")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise OcrError("input is not a decodable image")
    return arr, fmt


def load_rgb(image) -> np.ndarray:
    """Resolve and decode an `Image` envelope into an (H, W, 3) uint8 RGB array.

    Enforces the byte cap on the encoded input and the pixel cap on the
    header-declared dimensions BEFORE decoding, so a small 'bomb' image with
    huge declared dimensions is rejected without allocating its pixels.
    """
    arr, _fmt = _decode(_raw_bytes(image))
    return arr


def load_rgb_and_format(image):
    """Like `load_rgb`, but also returns the detected source format — needed
    by nodes (CorrectOrientation) that must re-encode and return image bytes,
    so the output is encoded in the same format the caller sent."""
    return _decode(_raw_bytes(image))


def encode_rgb(arr: np.ndarray, fmt: str) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array back to bytes in `fmt` (one of
    `_VALID_FORMATS`)."""
    pil = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


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


def correct_orientation(image) -> dict:
    """Classify whether the image is upside-down (0° vs 180°) using PP-OCRv4's
    angle-classification model (`RapidOCR.text_cls`, the same component the
    full pipeline runs before recognition) and, if so, rotate it 180° to
    correct it.

    We call `text_cls` directly (rather than the top-level `RapidOCR.__call__`
    with only `use_cls=True`) because the top-level wrapper discards the
    rotated pixels when detection is disabled — it only returns the label; the
    submodule is the documented way to get the corrected image back.

    Returns a dict: data (re-encoded bytes), format, width, height, rotated
    (bool), confidence (float in [0, 1]).
    """
    arr, fmt = load_rgb_and_format(image)
    engine = _engine()
    preprocessed, _ratio_h, _ratio_w = engine.preprocess(arr)
    _, cls_res, _ = engine.text_cls(preprocessed)
    label, score = cls_res[0]
    score = float(score)
    # Same decision rule PP-OCRv4's own pipeline applies internally
    # (ch_ppocr_cls/text_cls.py): only act on a confident "180" verdict.
    rotated = ("180" in str(label)) and (score > engine.text_cls.cls_thresh)
    out_arr = cv2.rotate(arr, cv2.ROTATE_180) if rotated else arr
    h, w = out_arr.shape[:2]
    return {
        "data": encode_rgb(out_arr, fmt),
        "format": fmt,
        "width": w,
        "height": h,
        "rotated": rotated,
        "confidence": score,
    }


def recognize_region(data: bytes, url: str, region) -> dict:
    """Crop `region` (an axis-aligned x0/y0/x1/y1 rectangle, clamped to the
    image bounds) out of the image resolved from (data, url) and run
    classification + recognition on just that crop, skipping detection
    entirely (`use_det=False`) — the same per-box pipeline stage the full
    Recognize pipeline runs on each detected box, applied here to a
    caller-supplied box instead of a detected one.

    Returns a dict: text, confidence. Raises OcrError on a degenerate region
    (zero area after clamping) or an unresolvable/undecodable image.
    """
    arr = load_rgb(_ImageLike(data, url))
    h, w = arr.shape[:2]
    x0 = max(0, min(int(round(region.x0)), w))
    y0 = max(0, min(int(round(region.y0)), h))
    x1 = max(0, min(int(round(region.x1)), w))
    y1 = max(0, min(int(round(region.y1)), h))
    if x1 <= x0 or y1 <= y0:
        raise OcrError("region has zero area after clamping to the image bounds")
    crop = np.ascontiguousarray(arr[y0:y1, x0:x1])
    out = _engine()(crop, use_det=False, use_cls=True, use_rec=True)
    rec_res = out[0] if isinstance(out, tuple) else out
    if not rec_res:
        return {"text": "", "confidence": 0.0}
    text, score = rec_res[0][0], rec_res[0][1]
    return {"text": str(text), "confidence": float(score)}


class _ImageLike:
    """Minimal (data, url) pair matching the shape `_raw_bytes`/`load_rgb`
    expect — lets `recognize_region` reuse them without a proto `Image`
    wrapper, since `RecognizeRegionInput` carries `data`/`url` as its own
    top-level fields (see messages.proto for why they're flattened rather
    than a nested `Image`)."""

    def __init__(self, data: bytes, url: str):
        self.data = data
        self.url = url
