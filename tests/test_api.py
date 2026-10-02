from io import BytesIO
import base64
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
FIXTURES = Path(__file__).parent / "fixtures"
REALISTIC_FIXTURES = FIXTURES / "realistic_men"
REALISTIC_WOMEN_FIXTURES = FIXTURES / "realistic_women"


def png_bytes(draw_fn):
    image = Image.new("RGB", (420, 260), "white")
    draw = ImageDraw.Draw(image)
    draw_fn(draw)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_page_exposes_all_workflows():
    response = client.get("/")
    assert response.status_code == 200
    assert "/api/v1/measure/compare" in response.text
    assert "/api/v1/measure/coin" in response.text
    assert response.text.count('class="preview"') == 2
    assert response.text.count('class="json-output"') == 2
    assert response.text.count('class="image-title"') == 2
    assert "annotation-hotspot" in response.text
    assert "renderHotspots" in response.text
    assert "/api/v1/measure/interactive-compare" in response.text
    assert response.text.count('class="tab-button"') == 3
    assert response.text.count('class="interactive-line"') == 1
    assert "pointermove" in response.text
    assert ".interactive-stage[hidden],.interactive-overlay[hidden]{display:none!important}" in response.text
    assert "relativePercentage:mix(lower.relative_percentage,upper.relative_percentage)" in response.text
    assert "Current relative thickness:" in response.text
    assert "showInteractiveOverlay=()=>svg.removeAttribute('hidden')" in response.text
    assert "hideInteractiveOverlay=()=>svg.setAttribute('hidden','')" in response.text
    assert "svg.hidden=" not in response.text
    assert 'markerUnits="userSpaceOnUse"' in response.text
    assert 'markerWidth="18" markerHeight="18"' in response.text
    assert 'annotated-preview' not in response.text


def test_compare_reports_larger_limb_and_area_percentage():
    data = png_bytes(lambda d: (d.rectangle((55, 30, 145, 225), fill=(210, 150, 100)), d.rectangle((260, 55, 330, 215), fill=(210, 150, 100))))
    response = client.post("/api/v1/measure/compare", files={"image": ("legs.png", data, "image/png")})
    body = response.json()
    assert response.status_code == 200
    assert body["larger_limb"] == "left"
    assert 45 < body["percentage_oversize"] < 70
    assert body["comparison_metric"] == "visible_projected_area"


def test_compare_preserves_left_right_position_when_right_is_larger():
    data = png_bytes(lambda d: (d.rectangle((60, 55, 125, 215), fill=(210, 150, 100)), d.rectangle((250, 25, 350, 225), fill=(210, 150, 100))))
    response = client.post("/api/v1/measure/compare", files={"image": ("arms.png", data, "image/png")})
    assert response.status_code == 200
    assert response.json()["larger_limb"] == "right"


def test_compare_returns_two_diameter_lines_and_ratio_overlay():
    data = png_bytes(lambda d: (d.rectangle((55, 30, 145, 225), fill=(210, 150, 100)), d.rectangle((260, 55, 330, 215), fill=(210, 150, 100))))
    response = client.post("/api/v1/measure/compare", files={"image": ("legs.png", data, "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    comparison = body["diameter_comparison"]
    assert comparison["metric"] == "visible_transverse_diameter"
    assert comparison["ratio_larger_to_smaller"] > 1
    assert [line["label"] for line in comparison["lines"]] == ["left", "right"]
    assert comparison["reference_percentage"] == 100
    assert comparison["reference_limb"] != comparison["larger_limb"]
    assert len(comparison["diameter_lines"]) == 2
    assert {line["position"] for line in comparison["diameter_lines"]} == {"upper", "lower"}
    assert {line["label"] for line in comparison["diameter_lines"]} == {comparison["larger_limb"]}
    assert all(line["relative_percentage"] >= 100 for line in comparison["diameter_lines"])
    assert comparison["lines"][0]["start_px"]["y"] == comparison["lines"][1]["start_px"]["y"]
    for line in comparison["lines"]:
        start, end, box = line["start_px"], line["end_px"], line["bounding_box_px"]
        assert line["line_width_px"] >= 10
        assert line["diameter_px"] > 0
        assert box == {
            "x": min(start["x"], end["x"]),
            "y": min(start["y"], end["y"]),
            "width": abs(end["x"] - start["x"]) + 1,
            "height": abs(end["y"] - start["y"]) + 1,
        }
    prefix, encoded = body["annotated_image"].split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as annotated:
        pixels = np.asarray(annotated.convert("RGB"))
    red_pixels = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 100) & (pixels[:, :, 2] < 120)
    assert int(red_pixels.sum()) > 40


def test_compare_line_labels_show_only_percentage_oversize():
    data = png_bytes(lambda d: (d.rectangle((55, 30, 145, 225), fill=(210, 150, 100)), d.rectangle((260, 55, 330, 215), fill=(210, 150, 100))))
    response = client.post("/api/v1/measure/compare", files={"image": ("legs.png", data, "image/png")})
    assert response.status_code == 200, response.text
    comparison = response.json()["diameter_comparison"]
    lines = comparison["diameter_lines"]
    assert len(lines) == 2
    for line in lines:
        assert line["length_cm"] is None
        assert line["length_label"] == f"{line['relative_percentage']}%"
        assert line["tooltip"] == line["length_label"]
        assert "." not in line["length_label"]
        assert "px" not in line["length_label"]
        assert "cm unavailable" not in line["tooltip"]
        assert line["label_bbox_px"]["width"] > 0
        assert line["label_bbox_px"]["height"] >= 30


@pytest.mark.parametrize("path", [
    REALISTIC_FIXTURES / "arms_02.png",
    REALISTIC_WOMEN_FIXTURES / "arms_02.png",
])
def test_realistic_compare_lines_stay_inside_each_arm(path):
    response = client.post("/api/v1/measure/compare", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    image = Image.open(path)
    image_width, image_height = image.size
    for line in body["diameter_comparison"]["lines"]:
        start, end = line["start_px"], line["end_px"]
        assert 0 < min(start["x"], end["x"]) < image_width
        assert max(start["x"], end["x"]) < image_width
        assert 0 <= min(start["y"], end["y"]) < image_height
        assert max(start["y"], end["y"]) < image_height
        assert line["diameter_px"] < image_width * 0.45
    annotated_lines = body["diameter_comparison"]["diameter_lines"]
    assert len(annotated_lines) == 2
    assert {line["label"] for line in annotated_lines} == {body["diameter_comparison"]["larger_limb"]}


def test_realistic_arms_01_diameter_lines_match_reference_values():
    path = REALISTIC_FIXTURES / "arms_01.png"
    response = client.post("/api/v1/measure/compare", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    comparison = body["diameter_comparison"]
    assert comparison["larger_limb"] == "left"
    assert comparison["percentage_oversize"] == pytest.approx(33.33, abs=1.0)
    assert comparison["ratio_larger_to_smaller"] == pytest.approx(1.333, abs=0.02)
    assert len(comparison["diameter_lines"]) == 2
    expected = [(399, 356, 614, 356, 215), (954, 353, 1115, 353, 161)]
    for line, (x1, y1, x2, y2, diameter) in zip(comparison["lines"], expected):
        assert line["start_px"]["x"] == pytest.approx(x1, abs=8)
        assert line["start_px"]["y"] == pytest.approx(y1, abs=8)
        assert line["end_px"]["x"] == pytest.approx(x2, abs=8)
        assert line["end_px"]["y"] == pytest.approx(y2, abs=8)
        assert line["diameter_px"] == pytest.approx(diameter, abs=8)


def test_realistic_legs_01_uses_calf_cross_sections():
    path = REALISTIC_FIXTURES / "legs_01.png"
    response = client.post("/api/v1/measure/compare", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    comparison = response.json()["diameter_comparison"]
    assert comparison["larger_limb"] == "left"
    assert comparison["percentage_oversize"] == pytest.approx(33.77, abs=0.1)
    assert comparison["ratio_larger_to_smaller"] == pytest.approx(1.338, abs=0.002)
    assert comparison["metric"] == "visible_transverse_diameter"
    assert comparison["lines"] == [
        {
            "label": "left",
            "position": None,
            "start_px": {"x": 447, "y": 391},
            "end_px": {"x": 648, "y": 391},
            "bounding_box_px": {"x": 447, "y": 391, "width": 202, "height": 1},
            "orientation": "horizontal",
            "diameter_px": 201.0,
            "line_width_px": 12,
        },
        {
            "label": "right",
            "position": None,
            "start_px": {"x": 934, "y": 391},
            "end_px": {"x": 1084, "y": 391},
            "bounding_box_px": {"x": 934, "y": 391, "width": 151, "height": 1},
            "orientation": "horizontal",
            "diameter_px": 150.0,
            "line_width_px": 12,
        },
    ]


def test_realistic_legs_02_only_annotates_the_thicker_right_leg():
    path = REALISTIC_FIXTURES / "legs_02.png"
    response = client.post("/api/v1/measure/compare", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    comparison = body["diameter_comparison"]
    lines = sorted(comparison["diameter_lines"], key=lambda line: line["start_px"]["y"])

    assert comparison["larger_limb"] == "right"
    assert comparison["reference_limb"] == "left"
    assert comparison["reference_percentage"] == 100
    assert len(lines) == 2
    assert {line["label"] for line in lines} == {"right"}
    assert [line["position"] for line in lines] == ["upper", "lower"]
    assert lines[1]["start_px"]["y"] - lines[0]["start_px"]["y"] >= 180
    assert lines[0]["diameter_px"] != lines[1]["diameter_px"]
    assert lines[0]["relative_percentage"] == pytest.approx(133, abs=2)
    assert lines[0]["relative_percentage"] != lines[1]["relative_percentage"]
    for line in lines:
        assert line["length_label"] == f"{line['relative_percentage']}%"
        assert "." not in line["length_label"]

    _, encoded = body["annotated_image"].split(",", 1)
    with Image.open(BytesIO(base64.b64decode(encoded))) as annotated_image:
        annotated = np.asarray(annotated_image.convert("RGB"))
    with Image.open(path) as source_image:
        original = np.asarray(source_image.convert("RGB"))
    changed = np.any(annotated != original, axis=2)
    midpoint = changed.shape[1] // 2
    assert int(changed[:, :midpoint].sum()) == 0
    assert int(changed[:, midpoint:].sum()) > 100


def test_interactive_compare_returns_ten_fixed_measurements_for_dragging():
    path = REALISTIC_FIXTURES / "legs_02.png"
    response = client.post(
        "/api/v1/measure/interactive-compare",
        files={"image": (path.name, path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    measurements = body["measurements"]

    assert body["mode"] == "interactive_compare"
    assert body["thicker_limb"] == "right"
    assert body["reference_limb"] == "left"
    assert body["reference_percentage"] == 100
    assert body["measurement_count"] == 10
    assert body["initial_measurement_index"] in range(10)
    assert len(measurements) == 10
    assert [measurement["index"] for measurement in measurements] == list(range(10))
    assert [measurement["y_px"] for measurement in measurements] == sorted(
        {measurement["y_px"] for measurement in measurements}
    )
    for measurement in measurements:
        line = measurement["line"]
        assert line["label"] == "right"
        assert line["start_px"]["y"] == measurement["y_px"]
        assert line["end_px"]["y"] == measurement["y_px"]
        assert line["diameter_px"] == measurement["thicker_diameter_px"]
        assert measurement["percentage_oversize"] > 0
        assert measurement["relative_percentage"] >= 100
        assert measurement["relative_percentage"] == round(100 + measurement["percentage_oversize"])
    assert body["source_image"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("filename", ["arms_01.png", "legs_01.png", "legs_02.png"])
def test_interactive_compare_supports_realistic_arm_and_leg_fixtures(filename):
    path = REALISTIC_FIXTURES / filename
    response = client.post(
        "/api/v1/measure/interactive-compare",
        files={"image": (path.name, path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, (filename, response.text)
    assert len(response.json()["measurements"]) == 10


def test_interactive_synthetic_legs_use_bounded_cross_section_lines():
    path = FIXTURES / "legs_01.png"
    response = client.post(
        "/api/v1/measure/interactive-compare",
        files={"image": (path.name, path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    initial = body["measurements"][body["initial_measurement_index"]]
    line = initial["line"]

    assert body["thicker_limb"] == "left"
    assert line["label"] == "left"
    assert line["orientation"] == "horizontal"
    assert line["start_px"]["y"] == line["end_px"]["y"] == initial["y_px"]
    assert line["diameter_px"] == initial["thicker_diameter_px"]
    assert 100 <= line["diameter_px"] <= 140
    assert 80 <= line["start_px"]["x"] < line["end_px"]["x"] <= 230


def test_interactive_womens_arms_01_has_visible_initial_left_arm_geometry():
    path = REALISTIC_WOMEN_FIXTURES / "arms_01.png"
    response = client.post(
        "/api/v1/measure/interactive-compare",
        files={"image": (path.name, path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    initial = body["measurements"][body["initial_measurement_index"]]
    line = initial["line"]

    assert body["thicker_limb"] == "left"
    assert line["label"] == "left"
    assert 0 < line["start_px"]["x"] < line["end_px"]["x"] < body["image_width_px"] / 2
    assert line["start_px"]["y"] == line["end_px"]["y"] == initial["y_px"]
    assert initial["relative_percentage"] == pytest.approx(154, abs=2)


def test_coin_measure_calibrates_longest_dimension():
    data = png_bytes(lambda d: (d.rectangle((65, 25, 115, 225), fill=(210, 150, 100)), d.ellipse((245, 100, 284, 139), fill=(200, 160, 45))))
    response = client.post("/api/v1/measure/coin", files={"image": ("arm.png", data, "image/png")})
    body = response.json()
    assert response.status_code == 200
    assert 11.0 < body["limb"]["longest_dimension_cm"] < 13.5
    assert body["coin"]["reference_diameter_cm"] == 2.325


def test_realistic_arm_coin_estimates_hand_diameter_from_23_cm_coin():
    path = REALISTIC_FIXTURES / "arm_coin_01.png"
    response = client.post("/api/v1/measure/coin", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    # A 1 EUR coin is 2.325 cm across (2.3 cm rounded). In this fixture the
    # visible hand diameter is approximately 6.5 cm.
    assert body["limb"]["diameter_cm"] == pytest.approx(6.5, abs=0.6)
    assert body["coin"]["reference_diameter_cm"] == pytest.approx(2.3, abs=0.03)


@pytest.mark.parametrize("filename", ["arm_coin_01.png", "arm_coin_02.png"])
def test_realistic_arm_coin_fixtures_are_detected(filename):
    path = REALISTIC_FIXTURES / filename
    response = client.post("/api/v1/measure/coin", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    assert response.json()["coin"]["diameter_px"] > 0


def test_realistic_leg_coin_01_overlay_stays_on_leg():
    path = REALISTIC_FIXTURES / "leg_coin_01.png"
    response = client.post("/api/v1/measure/coin", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    line = response.json()["diameter_line"]
    assert line["orientation"] == "horizontal"
    assert line["start_px"]["x"] > 300
    assert line["end_px"]["x"] < 900
    assert line["start_px"]["y"] < 700
    assert line["diameter_px"] == pytest.approx(247, abs=20)


def test_realistic_leg_coin_02_overlay_stays_on_ankle():
    path = REALISTIC_FIXTURES / "leg_coin_02.png"
    response = client.post("/api/v1/measure/coin", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    line = response.json()["diameter_line"]
    assert line["orientation"] == "horizontal"
    assert 650 < line["start_px"]["x"] < 900
    assert line["end_px"]["x"] < 1200
    assert line["start_px"]["y"] == line["end_px"]["y"]
    assert 250 < line["start_px"]["y"] < 600
    assert line["diameter_px"] == pytest.approx(350, abs=60)


def test_realistic_horizontal_arm_reports_thickness_not_arm_length():
    path = REALISTIC_FIXTURES / "arm_coin_02.png"
    response = client.post("/api/v1/measure/coin", files={"image": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    # Expected from approximately 5 cm image thickness / 1.5 cm image coin
    # diameter, scaled by the real 1 EUR diameter of approximately 2.3 cm.
    assert response.json()["limb"]["diameter_cm"] == pytest.approx(7.6, abs=0.6)


def test_coin_measure_returns_red_diameter_overlay_without_modifying_original():
    path = REALISTIC_FIXTURES / "arm_coin_01.png"
    original = path.read_bytes()
    response = client.post("/api/v1/measure/coin", files={"image": (path.name, original, "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    prefix, encoded = body["annotated_image"].split(",", 1)
    assert prefix == "data:image/png;base64"
    diameter_line = body["diameter_line"]
    assert len(body["diameter_lines"]) == 2
    assert {line["position"] for line in body["diameter_lines"]} == {"upper", "lower"}
    for line in body["diameter_lines"]:
        assert line["length_cm"] is not None
        formatted = f"{line['length_cm']:.2f}".rstrip("0").rstrip(".")
        assert line["length_label"] == f"{formatted}cm"
        assert line["tooltip"] == f"Length: {line['length_label']}"
        assert line["label_bbox_px"]["width"] > 0
        assert line["label_bbox_px"]["height"] > 0
    start = diameter_line["start_px"]
    end = diameter_line["end_px"]
    box = diameter_line["bounding_box_px"]
    assert diameter_line["orientation"] in {"horizontal", "vertical"}
    assert diameter_line["line_width_px"] >= 10
    assert box == {
        "x": min(start["x"], end["x"]),
        "y": min(start["y"], end["y"]),
        "width": abs(end["x"] - start["x"]) + 1,
        "height": abs(end["y"] - start["y"]) + 1,
    }
    with Image.open(BytesIO(base64.b64decode(encoded))) as annotated:
        pixels = np.asarray(annotated.convert("RGB"))
        assert annotated.size == Image.open(path).size
    red_pixels = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 100) & (pixels[:, :, 2] < 120)
    assert int(red_pixels.sum()) > 20
    assert path.read_bytes() == original


def test_compare_requires_two_silhouettes():
    data = png_bytes(lambda d: d.rectangle((100, 40, 160, 220), fill=(210, 150, 100)))
    response = client.post("/api/v1/measure/compare", files={"image": ("one.png", data, "image/png")})
    assert response.status_code == 422
    assert "two separate" in response.json()["detail"]


def test_coin_requires_image_media_type():
    response = client.post("/api/v1/measure/coin", files={"image": ("x.txt", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_all_arm_and_leg_comparison_fixtures_are_supported():
    for path in sorted(FIXTURES.glob("arms_*.png")) + sorted(FIXTURES.glob("legs_*.png")):
        response = client.post("/api/v1/measure/compare", files={"image": (path.name, path.read_bytes(), "image/png")})
        assert response.status_code == 200, (path.name, response.text)
        body = response.json()
        assert body["larger_limb"] in {"left", "right"}
        assert body["percentage_oversize"] > 0
        assert len(body["diameter_comparison"]["diameter_lines"]) == 2


def test_all_coin_calibration_fixtures_are_supported():
    for path in sorted(FIXTURES.glob("*_coin_*.png")):
        response = client.post("/api/v1/measure/coin", files={"image": (path.name, path.read_bytes(), "image/png")})
        assert response.status_code == 200, (path.name, response.text)
        body = response.json()
        assert body["limb"]["longest_dimension_cm"] > 0
        assert body["limb"]["diameter_cm"] > 0
        assert body["coin"]["diameter_px"] > 0
        assert len(body["diameter_lines"]) == 2


def test_realistic_fixture_set_contains_eight_decodable_pngs():
    expected = {
        "arms_01.png", "arms_02.png", "legs_01.png", "legs_02.png",
        "arm_coin_01.png", "arm_coin_02.png", "leg_coin_01.png", "leg_coin_02.png",
    }
    actual = {path.name for path in REALISTIC_FIXTURES.glob("*.png")}
    assert actual == expected
    for path in sorted(REALISTIC_FIXTURES.glob("*.png")):
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 512 and image.height >= 512


def test_realistic_women_fixture_set_contains_six_decodable_pngs():
    expected = {
        "arms_01.png", "arms_02.png",
        "arm_coin_01.png", "arm_coin_02.png", "leg_coin_01.png", "leg_coin_02.png",
    }
    actual = {path.name for path in REALISTIC_WOMEN_FIXTURES.glob("*.png")}
    assert actual == expected
    for path in sorted(REALISTIC_WOMEN_FIXTURES.glob("*.png")):
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 512 and image.height >= 512
