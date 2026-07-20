from gen.messages_pb2 import Image, TextOut
from gen.axiom_context import AxiomContext
from nodes._ocr import recognize as _recognize, OcrError


def extract_text(ax: AxiomContext, input: Image) -> TextOut:
    """Recognize all text in an image and return it as a single plain string,
    with lines joined in reading order by newlines — a clean edge into text or
    NLP nodes downstream. No geometry or confidence is returned; use Recognize
    when you need those. Malformed, oversized (>20 MB or >40 MP), or unfetchable
    input returns a structured error instead of raising. Wraps the Apache-2.0
    RapidOCR / PP-OCRv4 engine.
    """
    try:
        r = _recognize(input)
    except OcrError as exc:
        return TextOut(error=str(exc))
    return TextOut(text=r["text"])
