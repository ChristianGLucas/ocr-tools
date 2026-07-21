from gen.christiangeorgelucas_image_tools_messages_pb2 import Image
from gen.messages_pb2 import OrientationResult
from gen.axiom_context import AxiomContext
from nodes._ocr import correct_orientation as _correct_orientation, OcrError


def correct_orientation(ax: AxiomContext, input: Image) -> OrientationResult:
    """Detect whether the image's text is upside-down (0 vs. 180 degrees,
    PP-OCRv4's angle-classification model) and, if so, rotate the image to
    correct it; otherwise return it unchanged. Returns the (possibly rotated)
    image re-encoded as bytes/format/width/height, whether a correction was
    applied, and the classifier's confidence in [0,1]. Best applied to a
    single text line or a tight region crop — the classifier is trained on
    line-height crops, so accuracy on a full multi-line page is lower but
    still directionally useful. Malformed, oversized (>20 MB or >40 MP), or
    unfetchable input returns a structured error instead of raising. Wraps
    the Apache-2.0 RapidOCR / PP-OCRv4 angle-classification stage.
    """
    try:
        r = _correct_orientation(input)
    except OcrError as exc:
        return OrientationResult(error=str(exc))
    return OrientationResult(
        data=r["data"],
        format=r["format"],
        width=r["width"],
        height=r["height"],
        rotated=r["rotated"],
        confidence=r["confidence"],
    )
