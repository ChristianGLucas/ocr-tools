import io

import pytest
from PIL import Image as PILImage

from gen.messages_pb2 import OrientationResult
from gen.christiangeorgelucas_image_tools_messages_pb2 import Image
from nodes.correct_orientation import correct_orientation
from nodes.recognize import recognize


def _rotate_180_bytes(raw: bytes) -> bytes:
    # INDEPENDENT ORACLE CONSTRUCTION: we rotate the fixture ourselves, with
    # PIL's own transpose (not RapidOCR, not cv2, not anything the node under
    # test uses), so the "this image is upside-down" ground truth comes from
    # outside the code path being tested.
    pil = PILImage.open(io.BytesIO(raw))
    rotated = pil.transpose(PILImage.ROTATE_180)
    buf = io.BytesIO()
    rotated.save(buf, format=pil.format or "PNG")
    return buf.getvalue()


def test_correct_orientation_leaves_upright_image_unchanged(ax, fixture_bytes):
    data = fixture_bytes("hello_world.png")
    result = correct_orientation(ax, Image(data=data))
    assert result.error == ""
    assert result.rotated is False
    assert result.width == 560
    assert result.height == 130
    # Re-decoding the (unrotated) output must still read "HELLO WORLD".
    reread = recognize(ax, Image(data=result.data))
    up = reread.text.upper()
    assert "HELLO" in up and "WORLD" in up


def test_correct_orientation_flips_upside_down_image(ax, fixture_bytes):
    upright = fixture_bytes("hello_world.png")
    flipped = _rotate_180_bytes(upright)  # ground truth: genuinely upside-down
    result = correct_orientation(ax, Image(data=flipped))
    assert result.error == ""
    assert result.rotated is True
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.9  # PP-OCRv4's own cls_thresh
    # dimensions are unchanged by a 180-degree rotation
    assert result.width == 560
    assert result.height == 130
    # INDEPENDENT ORACLE: the corrected image, fed back through Recognize
    # (a separate node/code path, only readable if the pixels were genuinely
    # rotated back to upright), must read "HELLO WORLD" again.
    reread = recognize(ax, Image(data=result.data))
    up = reread.text.upper()
    assert "HELLO" in up and "WORLD" in up


def test_correct_orientation_is_deterministic(ax, fixture_bytes):
    data = fixture_bytes("receipt.png")
    a = correct_orientation(ax, Image(data=data))
    b = correct_orientation(ax, Image(data=data))
    assert a.SerializeToString() == b.SerializeToString()


def test_correct_orientation_blank_image_not_rotated(ax, fixture_bytes):
    result = correct_orientation(ax, Image(data=fixture_bytes("blank.png")))
    assert result.error == ""
    assert result.rotated is False


def test_correct_orientation_malformed_input_returns_error(ax):
    result = correct_orientation(ax, Image(data=b"this is definitely not an image"))
    assert isinstance(result, OrientationResult)
    assert result.error != ""
    assert result.data == b""


def test_correct_orientation_empty_input_returns_error(ax):
    result = correct_orientation(ax, Image())
    assert result.error != ""


def test_correct_orientation_oversize_bytes_rejected(ax):
    result = correct_orientation(ax, Image(data=b"\x00" * (20 * 1024 * 1024 + 1)))
    assert "20 MB" in result.error


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://127.0.0.1/",  # loopback
        "http://10.0.0.5/img.png",  # private
    ],
)
def test_correct_orientation_ssrf_blocked(ax, url):
    result = correct_orientation(ax, Image(url=url))
    assert result.error != ""
    assert "non-public" in result.error
