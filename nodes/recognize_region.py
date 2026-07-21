from gen.messages_pb2 import RecognizeRegionInput, RegionText
from gen.axiom_context import AxiomContext
from nodes._ocr import recognize_region as _recognize_region, OcrError


def recognize_region(ax: AxiomContext, input: RecognizeRegionInput) -> RegionText:
    """Recognize text within a single caller-specified rectangular region of
    an image, skipping detection entirely — pairs directly with
    DetectRegions' output (feed a returned region straight back in) or any
    externally known field position (e.g. a fixed form field), without
    re-running detection over the whole image. The region is clamped to the
    image bounds; a region with zero area after clamping returns a
    structured error. Malformed, oversized (>20 MB or >40 MP), or unfetchable
    input, or a degenerate region, returns a structured error instead of
    raising. Wraps the Apache-2.0 RapidOCR / PP-OCRv4 classification +
    recognition stages on the caller-supplied crop.
    """
    try:
        r = _recognize_region(bytes(input.data), input.url, input.region)
    except OcrError as exc:
        return RegionText(error=str(exc))
    return RegionText(text=r["text"], confidence=r["confidence"])
