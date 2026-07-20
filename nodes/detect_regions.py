from gen.messages_pb2 import Image, RegionsOut, Region, Point
from gen.axiom_context import AxiomContext
from nodes._ocr import detect_regions as _detect_regions, OcrError


def detect_regions(ax: AxiomContext, input: Image) -> RegionsOut:
    """Locate text regions in an image without recognizing their contents — a
    fast 'where is the text' pass for layout analysis, cropping, or routing.
    Each region carries its axis-aligned bounding box (pixels, top-left origin)
    and the raw detection quadrilateral. Malformed, oversized (>20 MB or
    >40 MP), or unfetchable input returns a structured error instead of raising.
    Wraps the detection stage of the Apache-2.0 RapidOCR / PP-OCRv4 engine.
    """
    try:
        found = _detect_regions(input)
    except OcrError as exc:
        return RegionsOut(error=str(exc))
    regions = [
        Region(
            x0=rg["x0"],
            y0=rg["y0"],
            x1=rg["x1"],
            y1=rg["y1"],
            quad=[Point(x=px, y=py) for (px, py) in rg["quad"]],
        )
        for rg in found
    ]
    return RegionsOut(regions=regions, count=len(regions))
