from gen.messages_pb2 import OcrResult, TextLine, Point
from gen.christiangeorgelucas_image_tools_messages_pb2 import Image
from gen.axiom_context import AxiomContext
from nodes._ocr import recognize as _recognize, OcrError


def recognize(ax: AxiomContext, input: Image) -> OcrResult:
    """Detect and recognize all text in an image, returning the full text plus
    every line's recognized string, confidence, and geometry. Boxes are in
    pixels with a top-left origin; each line also carries the raw detection
    quadrilateral so rotated or skewed text keeps its exact polygon. Malformed,
    oversized (>20 MB or >40 MP), or unfetchable input returns a structured
    error instead of raising. Wraps the Apache-2.0 RapidOCR / PP-OCRv4 engine.
    """
    try:
        r = _recognize(input)
    except OcrError as exc:
        return OcrResult(error=str(exc))
    lines = [
        TextLine(
            text=ln["text"],
            confidence=ln["confidence"],
            x0=ln["x0"],
            y0=ln["y0"],
            x1=ln["x1"],
            y1=ln["y1"],
            quad=[Point(x=px, y=py) for (px, py) in ln["quad"]],
        )
        for ln in r["lines"]
    ]
    return OcrResult(
        text=r["text"],
        lines=lines,
        mean_confidence=r["mean_confidence"],
        line_count=r["line_count"],
    )
