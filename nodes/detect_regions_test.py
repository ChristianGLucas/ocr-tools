from gen.messages_pb2 import RegionsOut
from gen.christiangeorgelucas_image_tools_messages_pb2 import Image
from nodes.detect_regions import detect_regions


def test_detect_regions_finds_text(ax, fixture_bytes):
    result = detect_regions(ax, Image(data=fixture_bytes("receipt.png")))
    assert result.error == ""
    assert result.count >= 1
    assert result.count == len(result.regions)
    for rg in result.regions:
        # box is well-formed and within the 560x220 image
        assert 0 <= rg.x0 < rg.x1 <= 560
        assert 0 <= rg.y0 < rg.y1 <= 220
        assert len(rg.quad) == 4


def test_detect_regions_blank_image_has_none(ax, fixture_bytes):
    result = detect_regions(ax, Image(data=fixture_bytes("blank.png")))
    assert result.error == ""
    assert result.count == 0


def test_detect_regions_is_deterministic(ax, fixture_bytes):
    data = fixture_bytes("receipt.png")
    a = detect_regions(ax, Image(data=data))
    b = detect_regions(ax, Image(data=data))
    assert a.SerializeToString() == b.SerializeToString()


def test_detect_regions_malformed_returns_error(ax):
    result = detect_regions(ax, Image(data=b"\x89PNG not really"))
    assert isinstance(result, RegionsOut)
    assert result.error != ""
    assert result.count == 0


def test_detect_regions_ssrf_blocked(ax):
    result = detect_regions(ax, Image(url="http://metadata.google.internal/"))
    # resolves to a link-local / private address -> blocked (or unresolvable)
    assert result.error != ""
