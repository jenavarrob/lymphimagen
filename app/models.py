from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LimbResult(BaseModel):
    label: str
    area_px: int = Field(gt=0)
    bounding_box_px: BoundingBox


class CoinResult(BaseModel):
    diameter_px: float = Field(gt=0)
    reference_diameter_cm: float = 2.325


class MeasuredLimb(BaseModel):
    longest_dimension_px: int = Field(gt=0)
    longest_dimension_cm: float = Field(gt=0)
    diameter_px: int = Field(gt=0)
    diameter_cm: float = Field(gt=0)


class PixelPoint(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class PixelBoundingBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DiameterLine(BaseModel):
    label: str | None = None
    position: str | None = None
    start_px: PixelPoint
    end_px: PixelPoint
    bounding_box_px: PixelBoundingBox
    orientation: Literal["horizontal", "vertical", "diagonal"]
    diameter_px: float = Field(gt=0)
    line_width_px: int = Field(gt=0)


class AnnotatedDiameterLine(DiameterLine):
    length_cm: float | None = Field(default=None, gt=0)
    length_label: str
    tooltip: str
    label_bbox_px: PixelBoundingBox | None = None
    relative_percentage: int | None = Field(default=None, ge=100)


class DiameterComparison(BaseModel):
    metric: Literal["visible_transverse_diameter"]
    larger_limb: Literal["left", "right"]
    reference_limb: Literal["left", "right"]
    reference_percentage: Literal[100] = 100
    percentage_oversize: float
    ratio_larger_to_smaller: float = Field(gt=0)
    lines: list[DiameterLine]
    diameter_lines: list[AnnotatedDiameterLine]


class CompareResponse(BaseModel):
    mode: Literal["compare"] = "compare"
    larger_limb: str
    percentage_oversize: float
    comparison_metric: Literal["visible_projected_area"] = "visible_projected_area"
    limbs: list[LimbResult]
    diameter_comparison: DiameterComparison
    annotated_image: str
    warnings: list[str] = []


class InteractiveMeasurement(BaseModel):
    index: int = Field(ge=0, lt=10)
    y_px: int = Field(ge=0)
    line: DiameterLine
    thicker_diameter_px: float = Field(gt=0)
    reference_diameter_px: float = Field(gt=0)
    percentage_oversize: float = Field(ge=0)
    relative_percentage: int = Field(ge=100)


class InteractiveCompareResponse(BaseModel):
    mode: Literal["interactive_compare"] = "interactive_compare"
    thicker_limb: Literal["left", "right"]
    reference_limb: Literal["left", "right"]
    reference_percentage: Literal[100] = 100
    measurement_count: Literal[10] = 10
    initial_measurement_index: int = Field(ge=0, lt=10)
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    measurements: list[InteractiveMeasurement] = Field(min_length=10, max_length=10)
    source_image: str
    warnings: list[str] = []


class CoinMeasureResponse(BaseModel):
    mode: Literal["coin_calibrated"] = "coin_calibrated"
    limb: MeasuredLimb
    coin: CoinResult
    diameter_line: DiameterLine
    diameter_lines: list[AnnotatedDiameterLine]
    annotated_image: str
    warnings: list[str] = []
