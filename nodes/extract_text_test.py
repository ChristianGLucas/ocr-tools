from gen.messages_pb2 import TextOut
from gen.christiangeorgelucas_image_tools_messages_pb2 import Image
from nodes.extract_text import extract_text


def test_extract_text_reads_known_text(ax, fixture_bytes):
    # INDEPENDENT ORACLE against the text baked into hello_world.png.
    result = extract_text(ax, Image(data=fixture_bytes("hello_world.png")))
    assert result.error == ""
    up = result.text.upper()
    assert "HELLO" in up and "WORLD" in up


def test_extract_text_multiline_joined_with_newlines(ax, fixture_bytes):
    result = extract_text(ax, Image(data=fixture_bytes("receipt.png")))
    assert result.error == ""
    # two source lines -> at least one newline separating recognized lines
    assert "\n" in result.text
    joined = result.text.upper().replace(" ", "")
    for token in ("INVOICE", "TOTAL"):
        assert token in joined


def test_extract_text_blank_image_is_empty(ax, fixture_bytes):
    result = extract_text(ax, Image(data=fixture_bytes("blank.png")))
    assert result.error == ""
    assert result.text == ""


def test_extract_text_malformed_returns_error(ax):
    result = extract_text(ax, Image(data=b"not an image at all"))
    assert isinstance(result, TextOut)
    assert result.error != ""
    assert result.text == ""


def test_extract_text_ssrf_blocked(ax):
    result = extract_text(ax, Image(url="http://169.254.169.254/"))
    assert "non-public" in result.error
