import pytest

from gen.messages_pb2 import RecognizeRegionInput, Region, RegionText
from nodes.recognize_region import recognize_region

# Known geometry for nodes/fixtures/receipt.png (560x220), established by
# running DetectRegions against it: line 1 "INVOICE 2026" sits roughly at
# (31,37)-(308,71); line 2 "Total 42 USD" sits roughly at (29,94)-(284,130).
# Padded slightly so a genuine crop test isn't pixel-exact-fragile.
_LINE1 = Region(x0=15, y0=20, x1=330, y1=80)
_LINE2 = Region(x0=15, y0=85, x1=310, y1=140)


def test_recognize_region_reads_only_the_cropped_line(ax, fixture_bytes):
    # INDEPENDENT ORACLE + SCOPING PROOF: line 1's known tokens must appear,
    # and line 2's known tokens must NOT — proving the node genuinely
    # recognizes only the given crop, not the whole image regardless of box.
    data = fixture_bytes("receipt.png")
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=_LINE1))
    assert result.error == ""
    up = result.text.upper().replace(" ", "")
    assert "INVOICE" in up and "2026" in up
    assert "TOTAL" not in up and "42" not in up
    assert 0.0 <= result.confidence <= 1.0


def test_recognize_region_second_line(ax, fixture_bytes):
    data = fixture_bytes("receipt.png")
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=_LINE2))
    assert result.error == ""
    up = result.text.upper().replace(" ", "")
    assert "TOTAL" in up and "42" in up
    assert "INVOICE" not in up


def test_recognize_region_is_deterministic(ax, fixture_bytes):
    data = fixture_bytes("receipt.png")
    a = recognize_region(ax, RecognizeRegionInput(data=data, region=_LINE1))
    b = recognize_region(ax, RecognizeRegionInput(data=data, region=_LINE1))
    assert a.SerializeToString() == b.SerializeToString()


def test_recognize_region_clamps_out_of_bounds_box(ax, fixture_bytes):
    # A box that overshoots the image bounds is clamped, not rejected.
    data = fixture_bytes("receipt.png")
    huge = Region(x0=0, y0=0, x1=100000, y1=100000)
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=huge))
    assert result.error == ""


def test_recognize_region_blank_image_has_no_text(ax, fixture_bytes):
    data = fixture_bytes("blank.png")
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=Region(x0=0, y0=0, x1=50, y1=50)))
    assert result.error == ""
    assert result.text == ""


def test_recognize_region_degenerate_region_returns_error(ax, fixture_bytes):
    data = fixture_bytes("receipt.png")
    zero_area = Region(x0=100, y0=100, x1=100, y1=200)  # zero width
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=zero_area))
    assert isinstance(result, RegionText)
    assert result.error != ""
    assert result.text == ""


@pytest.mark.parametrize(
    "bad_region",
    [
        Region(x0=float("nan"), y0=0, x1=100, y1=100),
        Region(x0=0, y0=0, x1=float("inf"), y1=100),
        Region(x0=0, y0=float("-inf"), x1=100, y1=100),
    ],
)
def test_recognize_region_non_finite_coords_return_error(ax, fixture_bytes, bad_region):
    # REGRESSION (adversarial review finding): NaN/Infinity are legal
    # proto3-JSON encodings of a `double` field, so a well-behaved-but-
    # imperfect caller can send them over the JSON<->protobuf bridge. Must
    # return the documented structured error, not raise ValueError/
    # OverflowError out of int(round(...)).
    data = fixture_bytes("receipt.png")
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=bad_region))
    assert isinstance(result, RegionText)
    assert result.error != ""
    assert "finite" in result.error
    assert result.text == ""


def test_recognize_region_inverted_region_returns_error(ax, fixture_bytes):
    data = fixture_bytes("receipt.png")
    inverted = Region(x0=200, y0=100, x1=50, y1=150)  # x1 < x0
    result = recognize_region(ax, RecognizeRegionInput(data=data, region=inverted))
    assert result.error != ""


def test_recognize_region_malformed_input_returns_error(ax):
    result = recognize_region(
        ax, RecognizeRegionInput(data=b"not an image", region=Region(x0=0, y0=0, x1=10, y1=10))
    )
    assert result.error != ""
    assert result.text == ""


def test_recognize_region_empty_input_returns_error(ax):
    result = recognize_region(ax, RecognizeRegionInput(region=Region(x0=0, y0=0, x1=10, y1=10)))
    assert result.error != ""


def test_recognize_region_oversize_bytes_rejected(ax):
    result = recognize_region(
        ax,
        RecognizeRegionInput(
            data=b"\x00" * (20 * 1024 * 1024 + 1), region=Region(x0=0, y0=0, x1=10, y1=10)
        ),
    )
    assert "20 MB" in result.error


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://127.0.0.1/",  # loopback
        "http://10.0.0.5/img.png",  # private
    ],
)
def test_recognize_region_ssrf_blocked(ax, url):
    result = recognize_region(ax, RecognizeRegionInput(url=url, region=Region(x0=0, y0=0, x1=10, y1=10)))
    assert result.error != ""
    assert "non-public" in result.error
