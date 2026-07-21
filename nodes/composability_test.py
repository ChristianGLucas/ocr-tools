"""Local composability proof for the two new nodes, standing in for
`axiom flow preview` — a platform bug in this environment's `axiom dev`
(reproducible even on the pristine, previously-published 3-node baseline;
ticket filed) prevents booting the dev server needed for a real flow preview
here. These tests chain the actual node functions directly, in-process, the
same way a compiled flow would wire their outputs to the next node's inputs,
and assert on the end-to-end result — not just that each node runs alone.
"""
from gen.messages_pb2 import RecognizeRegionInput
from gen.christiangeorgelucas_image_tools_messages_pb2 import Image
from nodes.detect_regions import detect_regions
from nodes.recognize_region import recognize_region
from nodes.correct_orientation import correct_orientation
from nodes.recognize import recognize


def test_detect_regions_into_recognize_region_chain(ax, fixture_bytes):
    # DetectRegions -> RecognizeRegion: exactly the "Region" edge the two
    # nodes were designed to share (no adapter/reshaping needed).
    data = fixture_bytes("receipt.png")
    regions = detect_regions(ax, Image(data=data))
    assert regions.error == ""
    assert regions.count >= 2

    recovered = []
    for rg in regions.regions:
        out = recognize_region(ax, RecognizeRegionInput(data=data, region=rg))
        assert out.error == ""
        recovered.append(out.text)

    joined = " ".join(recovered).upper().replace(" ", "")
    for token in ("INVOICE", "2026", "TOTAL", "42"):
        assert token in joined, f"missing {token!r} in chained output {recovered!r}"


def test_correct_orientation_into_recognize_chain(ax, fixture_bytes):
    # CorrectOrientation's output fields (data/format/width/height) are the
    # same shape as image-tools' `Image` fields by design, so they must flow
    # straight into another ocr-tools node's `Image` input (a 4-field map,
    # the same kind of edge a flow.yaml compose step would perform).
    data = fixture_bytes("hello_world.png")
    corrected = correct_orientation(ax, Image(data=data))
    assert corrected.error == ""

    fed_forward = recognize(ax, Image(data=corrected.data, format=corrected.format))
    assert fed_forward.error == ""
    up = fed_forward.text.upper()
    assert "HELLO" in up and "WORLD" in up
