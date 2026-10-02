from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


class AnalysisError(ValueError):
    """An image is valid media but cannot support the requested analysis."""


@dataclass(frozen=True)
class Component:
    area: int
    x: int
    y: int
    width: int
    height: int
    perimeter: int

    @property
    def circularity(self) -> float:
        return 4 * math.pi * self.area / max(self.perimeter * self.perimeter, 1)

    @property
    def extent(self) -> float:
        """Fraction of the component bounding box occupied by foreground."""
        return self.area / max(self.width * self.height, 1)


class ImageAnalyzer:
    EUR_COIN_CM = 2.325

    def decode(self, data: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(data)) as image:
                image = image.convert("RGB")
                if image.width < 40 or image.height < 40:
                    raise AnalysisError("Image must be at least 40 by 40 pixels")
                return np.asarray(image, dtype=np.uint8)
        except UnidentifiedImageError as exc:
            raise AnalysisError("Uploaded file is not a supported image") from exc

    @staticmethod
    def _png_data_uri(image: np.ndarray) -> str:
        output = BytesIO()
        Image.fromarray(image).save(output, format="PNG", optimize=False)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _mask(self, image: np.ndarray) -> np.ndarray:
        # Estimate the background from a thin border. This works well for the
        # uncluttered, contrasting PoC images described in the architecture.
        h, w, _ = image.shape
        border = np.concatenate((image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]))
        background = np.median(border, axis=0)
        distance = np.linalg.norm(image.astype(float) - background, axis=2)
        mask = distance > max(22.0, float(np.percentile(distance, 65)) * 0.35)
        # Remove isolated one-pixel noise without an image-processing package.
        neighbours = sum(
            np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
        )
        return mask & (neighbours >= 2)

    def _components(self, mask: np.ndarray) -> list[Component]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        components: list[Component] = []
        for y0, x0 in zip(*np.where(mask & ~seen)):
            if seen[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            seen[y0, x0] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) < 50:
                continue
            ys = np.fromiter((p[0] for p in pixels), dtype=int)
            xs = np.fromiter((p[1] for p in pixels), dtype=int)
            pixel_set = set(pixels)
            perimeter = sum((y + dy, x + dx) not in pixel_set for y, x in pixels for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            components.append(Component(len(pixels), int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), perimeter))
        return sorted(components, key=lambda c: c.area, reverse=True)

    def _skin_components(self, image: np.ndarray) -> list[Component]:
        """Find skin-like regions separately from shadows and coin pixels."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # This is intentionally broad: skin tone varies, and the result is
        # still filtered by connected-component size below.
        skin = cv2.inRange(
            hsv,
            np.array((0, 20, 40), dtype=np.uint8),
            np.array((28, 230, 255), dtype=np.uint8),
        ) > 0
        skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        return self._components(skin)

    def _skin_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        skin = cv2.inRange(
            hsv,
            np.array((0, 20, 40), dtype=np.uint8),
            np.array((28, 230, 255), dtype=np.uint8),
        ) > 0
        return cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    def _comparison_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """Use a higher saturation threshold so neutral backgrounds are excluded."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        skin = cv2.inRange(
            hsv,
            np.array((0, 50, 40), dtype=np.uint8),
            np.array((28, 230, 255), dtype=np.uint8),
        ) > 0
        skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Fill small skin-colour gaps caused by texture, highlights, and hair
        # so a calf or forearm is measured as one continuous cross-section.
        return cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0

    def _comparison_diameter_line(
        self,
        image: np.ndarray,
        side: str,
        fallback_limb: Component,
    ) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
        """Find a local limb cross-section instead of spanning a connected background region."""
        mask = self._comparison_skin_mask(image)
        height, width = mask.shape
        split_x = width / 2
        candidates: list[tuple[int, int, int]] = []
        scan_start = max(0, int(height * 0.05))
        scan_end = min(height, max(scan_start + 1, int(height * 0.65)))
        maximum_reasonable_span = max(20, int(width * 0.45))
        for y in range(scan_start, scan_end):
            row = mask[y]
            changes = np.diff(np.r_[False, row, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            for start, end in zip(starts, ends):
                span = int(end - start)
                center = (int(start) + int(end) - 1) / 2
                on_side = center < split_x if side == "left" else center >= split_x
                if on_side and 8 <= span <= maximum_reasonable_span:
                    candidates.append((span, int(start), int(y)))
        if candidates:
            target = float(np.median([candidate[0] for candidate in candidates]))
            span, start, y = min(candidates, key=lambda candidate: abs(candidate[0] - target))
            return span, ((start, y), (start + span - 1, y))
        return self._visible_thickness_px(self._mask(image), fallback_limb)

    def _comparison_side_runs(self, image: np.ndarray, side: str) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
        mask = self._comparison_skin_mask(image)
        _, width = mask.shape
        split_x = width / 2
        maximum_reasonable_span = max(20, int(width * 0.45))
        runs_by_y: dict[int, list[tuple[int, int]]] = {}
        for y, row in enumerate(mask):
            changes = np.diff(np.r_[False, row, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            runs = []
            for start, end in zip(starts, ends):
                span = int(end - start)
                center = (int(start) + int(end) - 1) / 2
                on_side = center < split_x if side == "left" else center >= split_x
                if on_side and 8 <= span <= maximum_reasonable_span:
                    runs.append((span, int(start)))
            if runs:
                runs_by_y[y] = runs
        return mask, runs_by_y

    def _comparison_shared_diameter_lines(
        self,
        image: np.ndarray,
        left_limb: Component,
        right_limb: Component,
    ) -> tuple[tuple[int, tuple[tuple[int, int], tuple[int, int]]], ...]:
        """Return upper/lower lines at matching Y coordinates for both limbs."""
        left_primary = self._comparison_diameter_line(image, "left", left_limb)
        right_primary = self._comparison_diameter_line(image, "right", right_limb)
        mask, left_runs = self._comparison_side_runs(image, "left")
        _, right_runs = self._comparison_side_runs(image, "right")
        common_y = sorted(set(left_runs).intersection(right_runs))
        if len(common_y) < 2:
            return (left_primary, right_primary, left_primary, right_primary)

        def nearest(target: int, minimum_distance: int = 0, excluded: int | None = None) -> int:
            eligible = [
                y for y in common_y
                if y != excluded and (excluded is None or abs(y - excluded) >= minimum_distance)
            ]
            return min(eligible or common_y, key=lambda y: abs(y - target))

        primary_y = nearest(left_primary[1][0][1])
        vertical_offset = max(20, int(mask.shape[0] * 0.28))
        secondary_target = primary_y + vertical_offset
        if secondary_target > common_y[-1]:
            secondary_target = primary_y - vertical_offset
        secondary_y = nearest(secondary_target, max(20, int(mask.shape[0] * 0.18)), primary_y)

        def line_at(runs_by_y: dict[int, list[tuple[int, int]]], y: int) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
            span, start = max(runs_by_y[y], key=lambda run: run[0])
            return span, ((start, y), (start + span - 1, y))

        return line_at(left_runs, primary_y), line_at(right_runs, primary_y), line_at(left_runs, secondary_y), line_at(right_runs, secondary_y)

    def _localized_diameter_line(
        self,
        mask: np.ndarray,
        limb: Component,
    ) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
        """Select the largest plausible cross-section inside a localized limb component."""
        roi = mask[limb.y:limb.y + limb.height, limb.x:limb.x + limb.width]
        candidates: list[tuple[int, int, int]] = []
        # A side-view lower leg can have a wider foot than ankle, making the
        # overall component look horizontal. It still needs a horizontal
        # cross-section through the ankle shaft.
        use_horizontal_cross_section = limb.height >= limb.width or (
            limb.y == 0 and limb.height >= mask.shape[0] * 0.70 and limb.width > limb.height
        )
        if use_horizontal_cross_section:
            side_view_ankle = limb.y == 0 and limb.height >= mask.shape[0] * 0.70 and limb.width > limb.height
            start_row = max(0, int(limb.height * (0.20 if side_view_ankle else 0.05)))
            end_row = min(limb.height, max(start_row + 1, int(limb.height * 0.65)))
            maximum_span = max(20, int(mask.shape[1] * 0.45))
            for row_index in range(start_row, end_row):
                row = roi[row_index]
                changes = np.diff(np.r_[False, row, False].astype(np.int8))
                starts = np.flatnonzero(changes == 1)
                ends = np.flatnonzero(changes == -1)
                for start, end in zip(starts, ends):
                    span = int(end - start)
                    if 8 <= span <= maximum_span:
                        candidates.append((span, int(start), row_index))
            if candidates:
                target = float(np.percentile([candidate[0] for candidate in candidates], 75))
                span, start, row_index = min(candidates, key=lambda candidate: abs(candidate[0] - target))
                y = limb.y + row_index
                return span, ((limb.x + start, y), (limb.x + start + span - 1, y))
        else:
            start_column = max(0, int(limb.width * 0.05))
            end_column = min(limb.width, max(start_column + 1, int(limb.width * 0.95)))
            maximum_span = max(20, int(mask.shape[0] * 0.45))
            for column_index in range(start_column, end_column):
                column = roi[:, column_index]
                changes = np.diff(np.r_[False, column, False].astype(np.int8))
                starts = np.flatnonzero(changes == 1)
                ends = np.flatnonzero(changes == -1)
                for start, end in zip(starts, ends):
                    span = int(end - start)
                    if 8 <= span <= maximum_span:
                        candidates.append((span, column_index, int(start)))
            if candidates:
                target = float(np.percentile([candidate[0] for candidate in candidates], 75))
                span, column_index, start = min(candidates, key=lambda candidate: abs(candidate[0] - target))
                x = limb.x + column_index
                return span, ((x, limb.y + start), (x, limb.y + start + span - 1))
        return self._visible_thickness_px(mask, limb)

    def _cross_section_candidates(
        self,
        mask: np.ndarray,
        limb: Component,
        horizontal_line: bool,
    ) -> list[tuple[int, int, tuple[tuple[int, int], tuple[int, int]]]]:
        roi = mask[limb.y:limb.y + limb.height, limb.x:limb.x + limb.width]
        candidates = []
        maximum_span = max(20, int(max(mask.shape) * 0.45))
        if horizontal_line:
            for row_index, row in enumerate(roi):
                xs = np.flatnonzero(row)
                if len(xs):
                    span = int(xs[-1] - xs[0] + 1)
                    if 8 <= span <= maximum_span:
                        y = limb.y + row_index
                        candidates.append((span, y, ((limb.x + int(xs[0]), y), (limb.x + int(xs[-1]), y))))
        else:
            for column_index, column in enumerate(roi.T):
                ys = np.flatnonzero(column)
                if len(ys):
                    span = int(ys[-1] - ys[0] + 1)
                    if 8 <= span <= maximum_span:
                        x = limb.x + column_index
                        candidates.append((span, x, ((x, limb.y + int(ys[0])), (x, limb.y + int(ys[-1])))))
        return candidates

    def _secondary_diameter_line(
        self,
        mask: np.ndarray,
        limb: Component,
        primary_line: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        horizontal_line = primary_line[0][1] == primary_line[1][1]
        candidates = self._cross_section_candidates(mask, limb, horizontal_line)
        if len(candidates) < 2:
            return primary_line
        primary_axis = primary_line[0][1] if horizontal_line else primary_line[0][0]
        axes = [candidate[1] for candidate in candidates]
        minimum_axis, maximum_axis = min(axes), max(axes)
        axis_range = max(maximum_axis - minimum_axis, 1)
        target_fraction = 0.70 if primary_axis <= (minimum_axis + maximum_axis) / 2 else 0.30
        target = minimum_axis + axis_range * target_fraction
        minimum_distance = max(20, int(axis_range * 0.15))
        eligible = [candidate for candidate in candidates if abs(candidate[1] - primary_axis) >= minimum_distance]
        secondary = min(eligible or candidates, key=lambda candidate: abs(candidate[1] - target))
        return secondary[2]

    @staticmethod
    def _coin_limb_spans_background(limb: Component, image: np.ndarray) -> bool:
        height, width, _ = image.shape
        touches_side = limb.x <= 0 or limb.x + limb.width >= width
        return touches_side and limb.width >= width * 0.75 and limb.height >= height * 0.70

    def _visible_thickness_px(self, mask: np.ndarray, limb: Component) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
        """Estimate limb thickness, excluding length and hand/finger spread."""
        roi = mask[limb.y:limb.y + limb.height, limb.x:limb.x + limb.width]
        if limb.height >= limb.width:
            # A vertical limb's cross-sections are horizontal rows. The
            # maximum captures the palm/calf width in front-facing images.
            best_span = 0
            best_points = ((limb.x, limb.y), (limb.x + limb.width - 1, limb.y))
            for row_index, row in enumerate(roi):
                xs = np.flatnonzero(row)
                if len(xs) and int(xs[-1] - xs[0] + 1) > best_span:
                    best_span = int(xs[-1] - xs[0] + 1)
                    y = limb.y + row_index
                    best_points = ((limb.x + int(xs[0]), y), (limb.x + int(xs[-1]), y))
            return best_span or min(limb.width, limb.height), best_points

        # A horizontal hand/arm often has fingers that make its bounding box
        # much taller than the forearm shaft. A low percentile of vertical
        # column spans is a stable shaft-thickness estimate.
        spans: list[tuple[int, int, int]] = []
        for column_index, column in enumerate(roi.T):
            ys = np.flatnonzero(column)
            if len(ys):
                spans.append((int(ys[-1] - ys[0] + 1), column_index, int(ys[0])))
        if not spans:
            span = min(limb.width, limb.height)
            return span, ((limb.x, limb.y), (limb.x, limb.y + span - 1))
        target = float(np.percentile([span[0] for span in spans], 15))
        span, column_index, top = min(spans, key=lambda item: abs(item[0] - target))
        return span, ((limb.x + column_index, limb.y + top), (limb.x + column_index, limb.y + top + span - 1))

    @staticmethod
    def _diameter_line_width(image: Image.Image) -> int:
        """Choose a stroke that remains visible when the image is scaled in the GUI."""
        return max(10, min(image.width, image.height) // 80)

    @staticmethod
    def _diameter_line_metadata(
        line: tuple[tuple[int, int], tuple[int, int]],
        line_width_px: int,
        label: str | None = None,
        position: str | None = None,
        length_cm: float | None = None,
        label_bbox: dict | None = None,
        display_label: str | None = None,
        tooltip_text: str | None = None,
    ) -> dict:
        (x1, y1), (x2, y2) = line
        diameter_px = round(math.hypot(x2 - x1, y2 - y1), 2)
        if display_label is not None:
            length_label = display_label
            tooltip = tooltip_text or display_label
        elif length_cm is None:
            length_label = f"{diameter_px:.2f}px (unscaled)"
            tooltip = f"Visible diameter: {diameter_px:.2f}px; cm unavailable without an image scale."
        else:
            length_cm = round(length_cm, 2)
            formatted_cm = f"{length_cm:.2f}".rstrip("0").rstrip(".")
            length_label = f"{formatted_cm}cm"
            tooltip = f"Length: {length_label}"
        metadata = {
            "start_px": {"x": x1, "y": y1},
            "end_px": {"x": x2, "y": y2},
            "bounding_box_px": {
                "x": min(x1, x2),
                "y": min(y1, y2),
                "width": abs(x2 - x1) + 1,
                "height": abs(y2 - y1) + 1,
            },
            "orientation": "horizontal" if y1 == y2 else "vertical" if x1 == x2 else "diagonal",
            "diameter_px": diameter_px,
            "line_width_px": line_width_px,
            "length_cm": length_cm,
            "length_label": length_label,
            "tooltip": tooltip,
            "label_bbox_px": label_bbox,
        }
        if label is not None:
            metadata["label"] = label
        if position is not None:
            metadata["position"] = position
        return metadata

    @staticmethod
    def _annotation_font(image: Image.Image) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        size = max(32, min(72, min(image.width, image.height) // 12))
        for path in (
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _draw_diameter_label(
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        line: tuple[tuple[int, int], tuple[int, int]],
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        line_width: int,
    ) -> dict[str, int]:
        (x1, y1), (x2, y2) = line
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = max(4, line_width // 3)
        box_width = text_width + padding * 2
        box_height = text_height + padding * 2
        margin = max(6, line_width)

        if y1 == y2:
            box_x = round((x1 + x2) / 2 - box_width / 2)
            box_y = min(y1, y2) - box_height - margin
            if box_y < 0:
                box_y = max(y1, y2) + margin
        else:
            box_x = max(x1, x2) + margin
            box_y = round((y1 + y2) / 2 - box_height / 2)
            if box_x + box_width > image.width:
                box_x = min(x1, x2) - box_width - margin

        box_x = max(0, min(box_x, image.width - box_width))
        box_y = max(0, min(box_y, image.height - box_height))
        box = (box_x, box_y, box_x + box_width, box_y + box_height)
        draw.rounded_rectangle(
            box,
            radius=max(4, padding),
            fill=(115, 15, 27),
            outline=(255, 255, 255),
            width=max(2, line_width // 4),
        )
        draw.text(
            (box_x + padding, box_y + padding),
            text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(70, 0, 0),
        )
        return {
            "x": box_x,
            "y": box_y,
            "width": box_width,
            "height": box_height,
        }

    @staticmethod
    def _draw_diameter_arrow(
        draw: ImageDraw.ImageDraw,
        line: tuple[tuple[int, int], tuple[int, int]],
        width: int,
        color: tuple[int, int, int] = (220, 35, 48),
    ) -> None:
        (x1, y1), (x2, y2) = line
        outline = (255, 255, 255)
        outline_width = width + max(6, width // 2)
        draw.line((x1, y1, x2, y2), fill=outline, width=outline_width)
        draw.line((x1, y1, x2, y2), fill=color, width=width)
        dx, dy = x2 - x1, y2 - y1
        length = max(math.hypot(dx, dy), 1.0)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        head = max(24, width * 4)
        for x, y, direction in ((x1, y1, 1), (x2, y2, -1)):
            tip = (x, y)
            back = (x + direction * ux * head, y + direction * uy * head)
            left = (back[0] + px * head * 0.55, back[1] + py * head * 0.55)
            right = (back[0] - px * head * 0.55, back[1] - py * head * 0.55)
            outline_back = (x + direction * ux * (head + 6), y + direction * uy * (head + 6))
            outline_left = (outline_back[0] + px * (head + 6) * 0.65, outline_back[1] + py * (head + 6) * 0.65)
            outline_right = (outline_back[0] - px * (head + 6) * 0.65, outline_back[1] - py * (head + 6) * 0.65)
            draw.polygon((tip, outline_left, outline_right), fill=outline)
            draw.polygon((tip, left, right), fill=color)

    def _annotate_diameters(
        self,
        data: bytes,
        lines: list[tuple[str | None, tuple[tuple[int, int], tuple[int, int]], str | None, float | None, str | None]],
    ) -> tuple[str, list[dict]]:
        """Return a PNG data URI with red diameter arrows and readable labels."""
        with Image.open(BytesIO(data)) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width = self._diameter_line_width(image)
        font = self._annotation_font(image)
        color = (220, 35, 48)
        prepared = []
        for label, line, position, length_cm, display_label in lines:
            self._draw_diameter_arrow(draw, line, width, color)
            diameter_px = round(math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]), 2)
            if display_label is not None:
                label_text = display_label
            elif length_cm is None:
                label_text = f"{diameter_px:.2f}px (unscaled)"
            else:
                rounded_cm = f"{length_cm:.2f}".rstrip("0").rstrip(".")
                label_text = f"{rounded_cm}cm"
            prepared.append((label, line, position, length_cm, label_text, display_label))

        metadata = []
        for label, line, position, length_cm, label_text, display_label in prepared:
            label_bbox = self._draw_diameter_label(draw, image, line, label_text, font, width)
            metadata.append(
                self._diameter_line_metadata(
                    line,
                    width,
                    label,
                    position,
                    length_cm,
                    label_bbox,
                    display_label,
                    display_label,
                )
            )
        output = BytesIO()
        image.save(output, format="PNG", optimize=False)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}", metadata

    def _annotate_diameter(self, data: bytes, line: tuple[tuple[int, int], tuple[int, int]]) -> tuple[str, dict]:
        """Return a separate PNG data URI and the rendered arrow geometry."""
        annotated_image, metadata = self._annotate_diameters(data, [(None, line, None, None, None)])
        return annotated_image, metadata[0]

    def _find_coin_component(self, image: np.ndarray, components: list[Component]) -> tuple[Component | None, float | None]:
        """Return a segmented coin or a Hough-detected circle diameter."""
        shape_candidates = [
            c for c in components
            if c.area >= 500
            and 0.60 <= c.width / c.height <= 1.67
            and c.extent >= 0.45
            and c.circularity >= 0.08
        ]
        if len(shape_candidates) > 1:
            largest_candidate = max(shape_candidates, key=lambda c: c.area)
            smaller_candidates = [c for c in shape_candidates if c.area <= largest_candidate.area * 0.20]
            shape_candidates = smaller_candidates or shape_candidates
        if shape_candidates:
            coin = min(shape_candidates, key=lambda c: c.area)
            return coin, (coin.width + coin.height) / 2

        # Photorealistic coins often merge with shadows or a textured floor in
        # the foreground mask. HoughCircles uses the circular boundary instead
        # of requiring a clean connected component.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        min_dimension = min(gray.shape)
        for threshold in (45, 35):
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(20, min_dimension // 20),
                param1=100,
                param2=threshold,
                minRadius=max(8, min_dimension // 120),
                maxRadius=max(20, min_dimension // 3),
            )
            if circles is not None and len(circles[0]):
                # Higher accumulator thresholds are preferred; the first
                # circle is the strongest candidate for the isolated coin.
                return None, float(circles[0][0][2] * 2)
        return None, None

    def interactive_compare(self, data: bytes) -> dict:
        """Build ten fixed paired measurements for a draggable single-line overlay."""
        image = self.decode(data)
        if len(self._components(self._mask(image))) < 2:
            raise AnalysisError("Could not detect two separate limb silhouettes")

        _, left_runs = self._comparison_side_runs(image, "left")
        _, right_runs = self._comparison_side_runs(image, "right")
        height, width = image.shape[:2]
        common_y = sorted(
            y for y in set(left_runs).intersection(right_runs)
            if int(height * 0.08) <= y <= int(height * 0.62)
        )
        if len(common_y) < 10:
            raise AnalysisError("Could not find ten shared measurement levels across both limbs")

        rows: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for y in common_y:
            left_run = max(left_runs[y], key=lambda run: run[0])
            right_run = max(right_runs[y], key=lambda run: run[0])
            rows.append((y, left_run, right_run))

        left_median = float(np.median([row[1][0] for row in rows]))
        right_median = float(np.median([row[2][0] for row in rows]))
        thicker_limb = "left" if left_median >= right_median else "right"
        reference_limb = "right" if thicker_limb == "left" else "left"

        thicker_index = 1 if thicker_limb == "left" else 2
        reference_index = 2 if thicker_limb == "left" else 1
        thicker_median = left_median if thicker_limb == "left" else right_median
        reference_median = right_median if thicker_limb == "left" else left_median
        usable_rows = [
            row for row in rows
            if row[thicker_index][0] > row[reference_index][0]
            and row[thicker_index][0] <= thicker_median * 1.75
            and row[reference_index][0] <= reference_median * 1.75
        ]
        if len(usable_rows) < 10:
            raise AnalysisError("Could not find ten levels where the thicker limb exceeds the reference limb")

        sample_indexes = [int(round(value)) for value in np.linspace(0, len(usable_rows) - 1, 10)]
        sampled_rows = [usable_rows[index] for index in sample_indexes]
        line_width = max(10, min(height, width) // 80)
        measurements = []
        for index, row in enumerate(sampled_rows):
            y = row[0]
            thicker_span, thicker_start = row[thicker_index]
            reference_span, _ = row[reference_index]
            thicker_diameter = max(1, thicker_span - 1)
            reference_diameter = max(1, reference_span - 1)
            percentage_oversize = (thicker_diameter - reference_diameter) / reference_diameter * 100
            line = ((thicker_start, y), (thicker_start + thicker_span - 1, y))
            measurements.append(
                {
                    "index": index,
                    "y_px": y,
                    "line": self._diameter_line_metadata(line, line_width, thicker_limb),
                    "thicker_diameter_px": round(float(thicker_diameter), 2),
                    "reference_diameter_px": round(float(reference_diameter), 2),
                    "percentage_oversize": round(float(percentage_oversize), 2),
                    "relative_percentage": int(round(thicker_diameter / reference_diameter * 100)),
                }
            )

        return {
            "thicker_limb": thicker_limb,
            "reference_limb": reference_limb,
            "reference_percentage": 100,
            "measurement_count": 10,
            "initial_measurement_index": 4,
            "image_width_px": width,
            "image_height_px": height,
            "measurements": measurements,
            "source_image": self._png_data_uri(image),
            "warnings": [
                "Interactive percentages use visible transverse widths and are not clinical circumference measurements."
            ],
        }

    def compare(self, data: bytes) -> dict:
        image = self.decode(data)
        mask = self._mask(image)
        components = self._components(mask)
        if len(components) < 2:
            raise AnalysisError("Could not detect two separate limb silhouettes")
        selected = sorted(components[:2], key=lambda c: c.x)
        smaller, larger = sorted(selected, key=lambda c: c.area)
        percent = (larger.area - smaller.area) / smaller.area * 100
        left_line_data, right_line_data, left_secondary_data, right_secondary_data = self._comparison_shared_diameter_lines(image, selected[0], selected[1])
        left_diameter_px, left_line = left_line_data
        right_diameter_px, right_line = right_line_data
        left_secondary_px, left_secondary_line = left_secondary_data
        right_secondary_px, right_secondary_line = right_secondary_data
        smaller_diameter, larger_diameter = sorted((left_diameter_px, right_diameter_px))
        diameter_percent = (larger_diameter - smaller_diameter) / smaller_diameter * 100
        left_average = (left_diameter_px + left_secondary_px) / 2
        right_average = (right_diameter_px + right_secondary_px) / 2
        larger_diameter_limb = "left" if left_average >= right_average else "right"
        reference_limb = "right" if larger_diameter_limb == "left" else "left"
        if larger_diameter_limb == "left":
            thicker_lines = (left_line, left_secondary_line)
            thicker_diameters = (left_diameter_px, left_secondary_px)
            reference_diameters = (right_diameter_px, right_secondary_px)
        else:
            thicker_lines = (right_line, right_secondary_line)
            thicker_diameters = (right_diameter_px, right_secondary_px)
            reference_diameters = (left_diameter_px, left_secondary_px)
        relative_percentages = tuple(
            max(100, int(round(thicker / reference * 100)))
            for thicker, reference in zip(thicker_diameters, reference_diameters)
        )
        annotated_image, annotated_lines = self._annotate_diameters(
            data,
            [
                (larger_diameter_limb, line, position, None, f"{relative_percentage}%")
                for line, position, relative_percentage in zip(
                    thicker_lines,
                    ("upper", "lower"),
                    relative_percentages,
                )
            ],
        )
        for line, relative_percentage in zip(annotated_lines, relative_percentages):
            line["relative_percentage"] = relative_percentage
        line_width = max(10, min(image.shape[:2]) // 80)
        primary_lines = [
            self._diameter_line_metadata(left_line, line_width, "left"),
            self._diameter_line_metadata(right_line, line_width, "right"),
        ]
        return {
            "larger_limb": "left" if selected[0] is larger else "right",
            "percentage_oversize": round(percent, 2),
            "limbs": [
                {"label": "left", "area_px": selected[0].area, "bounding_box_px": {"width": selected[0].width, "height": selected[0].height}},
                {"label": "right", "area_px": selected[1].area, "bounding_box_px": {"width": selected[1].width, "height": selected[1].height}},
            ],
            "diameter_comparison": {
                "metric": "visible_transverse_diameter",
                "larger_limb": larger_diameter_limb,
                "reference_limb": reference_limb,
                "reference_percentage": 100,
                "percentage_oversize": round(diameter_percent, 2),
                "ratio_larger_to_smaller": round(larger_diameter / smaller_diameter, 3),
                "lines": primary_lines,
                "diameter_lines": annotated_lines,
            },
            "annotated_image": annotated_image,
            "warnings": ["PoC result uses visible projected area; it does not estimate circumference."],
        }

    def coin_measure(self, data: bytes) -> dict:
        image = self.decode(data)
        components = self._components(self._mask(image))
        coin_component, diameter_px = self._find_coin_component(image, components)
        if diameter_px is None:
            raise AnalysisError("Could not detect a 1 EUR coin; place it fully visible and separate from the limb")
        if coin_component is not None:
            limbs = [c for c in components if c is not coin_component]
        else:
            limbs = components
        if not limbs:
            raise AnalysisError("Could not detect a limb silhouette")
        # Skin segmentation prevents a horizontally photographed arm from
        # being joined to the pale background by shadows or floor texture.
        skin_mask = self._skin_mask(image)
        skin_limbs = self._components(skin_mask)
        limb = max(skin_limbs or limbs, key=lambda c: c.area)
        measurement_mask = skin_mask if skin_limbs else self._mask(image)
        if self._coin_limb_spans_background(limb, image):
            localized_mask = self._comparison_skin_mask(image)
            localized_limbs = self._components(localized_mask)
            if localized_limbs:
                limb = max(localized_limbs, key=lambda c: c.area)
                measurement_mask = localized_mask
        dimension_px = max(limb.width, limb.height)
        dimension_cm = dimension_px * self.EUR_COIN_CM / diameter_px
        visible_diameter_px, diameter_line = self._localized_diameter_line(measurement_mask, limb) if measurement_mask is not skin_mask else self._visible_thickness_px(measurement_mask, limb)
        visible_diameter_cm = visible_diameter_px * self.EUR_COIN_CM / diameter_px
        secondary_line = self._secondary_diameter_line(measurement_mask, limb, diameter_line)
        horizontal_line = diameter_line[0][1] == diameter_line[1][1]
        ordered_lines = sorted((diameter_line, secondary_line), key=lambda line: line[0][1] if horizontal_line else line[0][0])
        positions = ("upper", "lower") if horizontal_line else ("left", "right")
        scale_cm_per_px = self.EUR_COIN_CM / diameter_px
        annotated_image, annotated_lines = self._annotate_diameters(
            data,
            [
                (None, line, position, math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]) * scale_cm_per_px, None)
                for line, position in zip(ordered_lines, positions)
            ],
        )
        primary_index = ordered_lines.index(diameter_line)
        diameter_line_metadata = {key: value for key, value in annotated_lines[primary_index].items() if key != "position"}
        return {
            "limb": {
                "longest_dimension_px": dimension_px,
                "longest_dimension_cm": round(dimension_cm, 2),
                "diameter_px": visible_diameter_px,
                "diameter_cm": round(visible_diameter_cm, 2),
            },
            "coin": {"diameter_px": round(diameter_px, 2), "reference_diameter_cm": self.EUR_COIN_CM},
            "diameter_line": diameter_line_metadata,
            "diameter_lines": annotated_lines,
            "annotated_image": annotated_image,
            "warnings": ["PoC result is a visible cross-sectional thickness, not anatomical circumference."],
        }
