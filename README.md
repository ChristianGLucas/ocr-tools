# ocr-tools

Composable **optical character recognition** nodes for the Axiom marketplace.
Turn a scanned or rendered image into machine-readable text — with per-line
bounding boxes and recognition confidence — as stateless, deterministic,
single-input → single-output functions.

Built for the Axiom marketplace under the handle `christiangeorgelucas`.

## Nodes

All three nodes consume the single canonical **`Image`** envelope (raw `data`
bytes, or an `url` to fetch) and return a purpose-shaped result. The `Image`
message is **imported directly** from
[`christiangeorgelucas/image-tools`](https://github.com/ChristianGLucas/image-tools)
(it is the same type, not a copy), so image preprocessing composes into OCR with
no adapter — e.g. `image-tools/Load → ocr-tools/Recognize` — and both packages'
nodes share one `Image` type inside a compiled flow. OCR text then flows on into
text or PDF-style downstream nodes.

| Node | Input → Output | What it does |
|---|---|---|
| **Recognize** | `Image → OcrResult` | Detect + recognize all text. Returns the full text, and for each line its string, confidence, axis-aligned box, and raw detection quadrilateral, plus mean confidence and line count. |
| **ExtractText** | `Image → TextOut` | Recognize all text and return it as one plain string (lines joined by newlines) — a clean edge into text/NLP nodes. |
| **DetectRegions** | `Image → RegionsOut` | Detection only (no recognition): a fast "where is the text" pass returning region boxes + quads. |

### Coordinates

Bounding boxes are in **pixels, origin top-left** (the raster convention). Note
this differs from `pdf-tools`, whose boxes are in PDF points with a bottom-left
origin — the two operate on different media and are deliberately not unified.

### Robustness

- **Offline & deterministic.** The OCR models ship inside the package; there is
  no network access at inference time and no model download. Repeated calls on
  the same input return byte-identical output.
- **Bounded input.** Encoded images are capped at 20 MB and decoded images at 40
  megapixels (checked against declared dimensions before decoding, so a
  decompression-bomb image is rejected without allocating its pixels).
- **SSRF-guarded fetch.** When you pass a `url`, the fetch rejects non-http(s)
  schemes and any host resolving to a private, loopback, link-local, reserved,
  multicast, or unspecified address. The check is re-run on **every redirect
  hop**, and each request connects to the exact validated IP — so an open
  redirector or a DNS-rebinding flip cannot steer the fetch to an internal
  address.
- **No crashes.** Malformed, oversized, or unfetchable input returns a
  structured `error` field, never an exception.

## Engine & license

This package is MIT-licensed. It **vendors** the OCR engine —
[RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr-onnxruntime` 1.4.4)
with Baidu's PP-OCRv4 ONNX models — under `vendor/rapidocr_onnxruntime/`, which
is **Apache-2.0** (see `vendor/rapidocr_onnxruntime/LICENSE.txt`). Vendoring lets
us pin exact code and weights for reproducible offline inference and excludes
RapidOCR's declared-but-unused `tqdm` dependency, keeping the entire runtime
dependency closure permissive (MIT / BSD / Apache-2.0 / HPND):

`onnxruntime` (MIT) · `opencv-python-headless` (Apache-2.0) · `numpy` (BSD) ·
`pyclipper` (MIT) · `shapely` (BSD) · `Pillow` (HPND) · `PyYAML` (MIT) ·
`six` (MIT).
