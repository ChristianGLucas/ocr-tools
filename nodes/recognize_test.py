import pytest

from gen.messages_pb2 import Image, OcrResult
from nodes.recognize import recognize
from nodes._ocr import _validate_url, OcrError


def test_validate_url_allows_public_numeric_ip():
    # A public address validates and is pinned (no DNS needed for a numeric IP).
    scheme, host, port, ip = _validate_url("http://8.8.8.8/image.png")
    assert scheme == "http"
    assert host == "8.8.8.8"
    assert port == 80
    assert ip == "8.8.8.8"


@pytest.mark.parametrize(
    "url,msg",
    [
        ("http://169.254.169.254/latest/meta-data/", "non-public"),  # cloud metadata
        ("http://127.0.0.1/", "non-public"),  # loopback
        ("http://10.0.0.5/x.png", "non-public"),  # private
        ("http://[::1]/", "non-public"),  # IPv6 loopback
        ("http://192.168.1.1/", "non-public"),  # private
        ("file:///etc/passwd", "http"),  # non-http scheme
        ("ftp://8.8.8.8/x", "http"),  # non-http scheme
    ],
)
def test_validate_url_blocks_disallowed(url, msg):
    # _validate_url is re-run on the initial URL AND every redirect hop, so this
    # is the guard that also blocks a public URL redirecting to an internal one.
    with pytest.raises(OcrError) as exc:
        _validate_url(url)
    assert msg in str(exc.value)


def test_recognize_reads_known_text(ax, fixture_bytes):
    # INDEPENDENT ORACLE: hello_world.png has the text "HELLO WORLD" baked in at
    # authoring time. Recovering those tokens proves recognition against ground
    # truth that does not come from RapidOCR itself.
    result = recognize(ax, Image(data=fixture_bytes("hello_world.png")))
    assert result.error == ""
    assert result.line_count >= 1
    assert result.line_count == len(result.lines)
    up = result.text.upper()
    assert "HELLO" in up and "WORLD" in up
    # confidences are real probabilities and the mean is meaningful
    for ln in result.lines:
        assert 0.0 <= ln.confidence <= 1.0
    assert result.mean_confidence > 0.5
    # geometry: every box is within the 560x130 image and well-formed
    for ln in result.lines:
        assert 0 <= ln.x0 < ln.x1 <= 560
        assert 0 <= ln.y0 < ln.y1 <= 130
        assert len(ln.quad) == 4


def test_recognize_multiline_oracle(ax, fixture_bytes):
    # receipt.png has two known lines. All four distinctive tokens must appear.
    result = recognize(ax, Image(data=fixture_bytes("receipt.png")))
    assert result.error == ""
    joined = result.text.upper().replace(" ", "")
    for token in ("INVOICE", "2026", "TOTAL", "42"):
        assert token in joined, f"missing {token!r} in {result.text!r}"


def test_recognize_is_deterministic(ax, fixture_bytes):
    data = fixture_bytes("hello_world.png")
    a = recognize(ax, Image(data=data))
    b = recognize(ax, Image(data=data))
    # byte-for-byte identical output on identical input
    assert a.SerializeToString() == b.SerializeToString()


def test_recognize_blank_image_has_no_text(ax, fixture_bytes):
    result = recognize(ax, Image(data=fixture_bytes("blank.png")))
    assert result.error == ""
    assert result.line_count == 0
    assert result.text == ""
    assert result.mean_confidence == 0.0


def test_recognize_malformed_input_returns_error(ax):
    result = recognize(ax, Image(data=b"this is definitely not an image"))
    assert isinstance(result, OcrResult)
    assert result.error != ""
    assert result.line_count == 0


def test_recognize_empty_input_returns_error(ax):
    result = recognize(ax, Image())
    assert result.error != ""


def test_recognize_oversize_bytes_rejected(ax):
    # 20 MB + 1 of non-image bytes: the byte cap must fire before any decode.
    result = recognize(ax, Image(data=b"\x00" * (20 * 1024 * 1024 + 1)))
    assert "20 MB" in result.error


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://127.0.0.1/",  # loopback
        "http://10.0.0.5/img.png",  # private
    ],
)
def test_recognize_ssrf_blocked(ax, url):
    result = recognize(ax, Image(url=url))
    assert result.error != ""
    assert "non-public" in result.error


def test_recognize_non_http_scheme_blocked(ax):
    result = recognize(ax, Image(url="file:///etc/passwd"))
    assert "http" in result.error
