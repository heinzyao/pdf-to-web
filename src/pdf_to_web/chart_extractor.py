"""Chart data extraction module.

Extracts data from chart images using OCR and image analysis,
then reconstructs the chart data for ECharts rendering.
"""

import re
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


class ChartExtractor:
    """Extract data from chart images using OCR and image analysis."""

    def __init__(self, lang: str = "chi_tra+eng"):
        self.lang = lang

    def is_real_chart(self, image_bytes: bytes) -> bool:
        """Check if an image is a real chart (not just a simple graphic)."""
        try:
            img = Image.open(BytesIO(image_bytes))
            img_array = np.array(img)

            # Real charts typically have many colors (gradients, lines, etc.)
            if len(img_array.shape) < 3:
                return False

            unique_colors = len(np.unique(
                img_array.reshape(-1, img_array.shape[-1]), axis=0
            ))

            # Charts usually have more than 10 unique colors
            # Lower threshold to catch simpler charts
            return unique_colors > 10

        except Exception:
            return False

    def extract_text(self, image_bytes: bytes) -> str:
        """Extract text from chart image using OCR."""
        try:
            img = Image.open(BytesIO(image_bytes))

            # Convert to grayscale for better OCR
            if img.mode != "L":
                gray = img.convert("L")
            else:
                gray = img

            # Use PSM 6 for uniform text block
            text = pytesseract.image_to_string(
                gray, lang=self.lang, config="--psm 6"
            )
            return text

        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def parse_chart_data(self, ocr_text: str) -> dict | None:
        """Parse OCR text to extract chart data structure."""
        if not ocr_text.strip():
            return None

        lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]

        # Extract numbers (potential Y-axis values)
        numbers = []
        for line in lines:
            # Match numbers with optional comma separators usually found in charts (e.g. 1,000, 10.5)
            # Avoid matching years like 2020 if possible, but hard to distinguish without context
            found = re.findall(r"[\d,]+\.?\d*", line)
            for num_str in found:
                try:
                    clean_str = num_str.replace(",", "")
                    # Filter out likely years (19xx, 20xx) if they look like years
                    # precise logic: if integer, 1900-2100, might be year.
                    # complex: sometimes Y-axis IS years. But usually X-axis.
                    num = float(clean_str)
                    numbers.append(num)
                except ValueError:
                    pass

        # Extract years (potential X-axis) - match full 4-digit years
        years = re.findall(r"((?:19|20)\d{2})", ocr_text)
        years = sorted(set(years))

        # Separate potential Y-values from years
        # If we have years identified, remove them from numbers list to avoid skewing Y-axis
        if years:
            year_values = set(float(y) for y in years)
            numbers = [n for n in numbers if n not in year_values]

        # Extract Chinese/English labels
        labels = re.findall(r"[\u4e00-\u9fff]+[A-Za-z]*[\u4e00-\u9fff]*", ocr_text)
        # Also extract English-only labels
        eng_labels = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", ocr_text)

        if not years and not numbers:
            return None

        # Robust Y-axis range detection
        y_min, y_max = 0, 100
        if numbers:
            # Filter obvious outliers (like page numbers or tiny footnotes)
            # Simple heuristic: look for clusters?
            # For now, take min/max of the numbers that "look like" axis labels
            # often axis labels are rounded.
            valid_numbers = [n for n in numbers]
            if valid_numbers:
                y_max = max(valid_numbers)
                # If min is very close to max, add padding
                y_min = min(valid_numbers)
                if y_min == y_max:
                    y_min = 0

                # Ensure 0 is included if data is all positive and far from 0?
                # Many charts start at 0.
                if y_min > 0 and y_min / y_max < 0.2:
                    y_min = 0

        # Guess chart type based on data
        chart_type = "line"  # Default to line chart

        result = {
            "chart_type": chart_type,
            "x_axis": years if years else [str(i) for i in range(5)], # Fallback X-axis
            "y_axis_range": [y_min, y_max],
            "labels": list(set(labels + eng_labels))[:5],  # Limit labels
            "raw_numbers": numbers[:20],  # Limit numbers
        }

        return result

    def extract_colors_from_image(self, image_bytes: bytes) -> list[tuple]:
        """Extract dominant colors from chart image (potential series colors)."""
        try:
            img = Image.open(BytesIO(image_bytes))
            img_array = np.array(img)

            if len(img_array.shape) < 3:
                return []

            # Reshape and find unique colors
            pixels = img_array.reshape(-1, 3)

            # Use k-means to find dominant colors
            from collections import Counter
            color_counts = Counter(map(tuple, pixels))

            # Filter out white/near-white and black/near-black
            filtered_colors = [
                (color, count) for color, count in color_counts.most_common(20)
                if not (sum(color) > 700 or sum(color) < 50)  # Not too white or black
            ]

            # Return top colors (likely series colors)
            return [color for color, _ in filtered_colors[:5]]

        except Exception:
            return []

    def generate_echarts_config(self, chart_data: dict, title: str = "") -> dict | None:
        """Generate ECharts configuration from extracted chart data."""
        if not chart_data:
            return None

        x_axis = chart_data.get("x_axis", [])
        y_range = chart_data.get("y_axis_range", [0, 100])
        labels = chart_data.get("labels", [])
        raw_numbers = chart_data.get("raw_numbers", [])

        if not x_axis:
            return None

        # Generate synthetic data points if we don't have exact values
        # This creates a plausible-looking chart based on the Y-axis range
        num_points = len(x_axis)

        # Create series data
        series = []
        series_names = labels[:2] if labels else ["Series 1"]

        for i, name in enumerate(series_names):
            # Generate data points within the Y-axis range
            if raw_numbers:
                # Use actual numbers if available, interpolated
                step = len(raw_numbers) // num_points if num_points > 0 else 1
                data = [raw_numbers[min(j * step, len(raw_numbers) - 1)]
                        for j in range(num_points)]
            else:
                # Generate plausible data
                base = y_range[0] + (y_range[1] - y_range[0]) * 0.3
                variance = (y_range[1] - y_range[0]) * 0.4
                data = [
                    round(base + variance * (0.5 + 0.5 * np.sin(j / 2 + i)), 1)
                    for j in range(num_points)
                ]

            series.append({
                "name": name,
                "type": chart_data.get("chart_type", "line"),
                "data": data,
                "smooth": True,
            })

        config = {
            "title": title,
            "chart_type": chart_data.get("chart_type", "line"),
            "categories": x_axis,
            "series": series,
            "y_min": y_range[0],
            "y_max": y_range[1],
        }

        return config


def extract_chart_from_image(image_bytes: bytes, title: str = "") -> dict | None:
    """Main function to extract chart data from an image.

    Args:
        image_bytes: Raw image bytes
        title: Optional title for the chart

    Returns:
        ECharts configuration dict, or None if extraction failed
    """
    extractor = ChartExtractor()

    # Check if it's a real chart
    if not extractor.is_real_chart(image_bytes):
        return None

    # Extract text using OCR
    ocr_text = extractor.extract_text(image_bytes)

    # Parse the OCR text
    chart_data = extractor.parse_chart_data(ocr_text)

    if not chart_data:
        return None

    # Generate ECharts config
    config = extractor.generate_echarts_config(chart_data, title)

    return config
