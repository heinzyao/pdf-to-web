"""Chart analysis and re-rendering module using OpenCV and matplotlib.

This module analyzes chart images using computer vision techniques
and re-renders them as clean, vector-quality charts using matplotlib.
"""

import io

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Configure Chinese font support
def _setup_chinese_font():
    """Setup Chinese font for matplotlib."""
    # Rebuild font cache to ensure fresh fonts
    fm._load_fontmanager(try_read_cache=False)

    # Try common CJK fonts on different systems
    chinese_fonts = [
        'Noto Sans CJK TC',
        'Noto Sans TC',
        'Heiti TC',
        'PingFang TC',
        'PingFang SC',
        'PingFang HK',
        'Heiti SC',
        'STHeiti',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font_name

    # Fallback: try to find any font with CJK support
    for font in fm.fontManager.ttflist:
        if 'cjk' in font.name.lower() or 'chinese' in font.name.lower():
            plt.rcParams['font.sans-serif'] = [font.name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font.name

    return None

_CJK_FONT = _setup_chinese_font()


class ChartAnalyzer:
    """Analyze chart images using computer vision."""

    def __init__(self):
        self.chart_colors = []
        self.data_points = []

    def load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def find_chart_area(self, img: np.ndarray) -> tuple:
        """Find the main chart plotting area (excluding axes and labels)."""
        h, w = img.shape[:2]

        # Estimate chart area (typically 10-90% of image)
        x_start = int(w * 0.1)
        x_end = int(w * 0.95)
        y_start = int(h * 0.05)
        y_end = int(h * 0.85)

        return x_start, y_start, x_end, y_end

    def extract_line_colors(self, img: np.ndarray) -> list:
        """Extract dominant colors that might be chart lines."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Find non-white, non-gray pixels (likely chart elements)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Mask for colored pixels (not too white, not too dark, has saturation)
        mask = (saturation > 30) & (value > 50) & (value < 250)

        if np.sum(mask) < 100:
            return []

        # Get colors from masked area
        colors = img[mask]

        # Use k-means to find dominant colors
        from collections import Counter
        color_counts = Counter(map(tuple, colors))

        # Get top colors (likely line/bar colors)
        top_colors = [color for color, _ in color_counts.most_common(5)]

        return top_colors

    def trace_line_data(self, img: np.ndarray, color: tuple, tolerance: int = 40) -> list:
        """Trace a colored line in the chart to extract data points."""
        h, w = img.shape[:2]
        x_start, y_start, x_end, y_end = self.find_chart_area(img)

        # Create color mask
        lower = np.array([max(0, c - tolerance) for c in color])
        upper = np.array([min(255, c + tolerance) for c in color])
        mask = cv2.inRange(img, lower, upper)

        # Find y-position for each x in chart area
        data_points = []
        for x in range(x_start, x_end, max(1, (x_end - x_start) // 50)):
            column = mask[y_start:y_end, x]
            y_positions = np.where(column > 0)[0]

            if len(y_positions) > 0:
                # Use median y position
                y = int(np.median(y_positions)) + y_start
                # Normalize to 0-1 range (inverted because y=0 is top)
                normalized_y = 1.0 - (y - y_start) / (y_end - y_start)
                normalized_x = (x - x_start) / (x_end - x_start)
                data_points.append((normalized_x, normalized_y))

        return data_points

    def detect_chart_type(self, img: np.ndarray) -> str:
        """Detect if chart is line, bar, or other type."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

        if lines is None:
            return "unknown"

        # Count horizontal vs vertical lines
        horizontal = 0
        vertical = 0
        diagonal = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 15 or angle > 165:
                horizontal += 1
            elif 75 < angle < 105:
                vertical += 1
            else:
                diagonal += 1

        # Bar charts have many vertical lines
        if vertical > horizontal * 2:
            return "bar"
        # Line charts have more diagonal/curved lines
        elif diagonal > vertical:
            return "line"
        else:
            return "line"  # Default to line

    def analyze(self, image_bytes: bytes) -> dict:
        """Full analysis of a chart image."""
        img = self.load_image(image_bytes)

        chart_type = self.detect_chart_type(img)
        colors = self.extract_line_colors(img)

        series_data = []
        for i, color in enumerate(colors[:3]):  # Max 3 series
            points = self.trace_line_data(img, color)
            if len(points) >= 3:  # Need at least 3 points
                series_data.append({
                    "color": color,
                    "points": points,
                    "name": f"Series {i+1}"
                })

        return {
            "chart_type": chart_type,
            "series": series_data,
            "width": img.shape[1],
            "height": img.shape[0]
        }


class ChartRenderer:
    """Re-render charts using matplotlib."""

    def __init__(self):
        self.fig_dpi = 150
        self.style = "seaborn-v0_8-whitegrid"

    def render_line_chart(self, analysis: dict, title: str = "",
                          x_labels: list = None, y_limits: tuple = None) -> bytes:
        """Render a line chart based on analysis."""
        plt.style.use(self.style)
        # Re-apply CJK font after style change
        if _CJK_FONT:
            plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(10, 5), dpi=self.fig_dpi)

        # Color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

        # Determine Y-axis scaling
        y_min, y_max = 0, 1
        if y_limits:
            y_min, y_max = y_limits

        y_range = y_max - y_min

        for i, series in enumerate(analysis.get("series", [])):
            points = series.get("points", [])
            if not points:
                continue

            x_vals = [p[0] for p in points]
            # Scale y-values from 0-1 to real range
            y_vals = [y_min + p[1] * y_range for p in points]

            color = colors[i % len(colors)]
            ax.plot(x_vals, y_vals, color=color, linewidth=2.5,
                    label=series.get("name", f"Series {i+1}"),
                    marker='o', markersize=4, alpha=0.9)

        # Style the chart
        ax.set_xlim(0, 1)
        ax.set_ylim(y_min, y_max)

        if x_labels:
            ax.set_xticks(np.linspace(0, 1, len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=9)
        else:
            ax.set_xticks([])

        # Show Y-axis labels if we have real limits
        if y_limits:
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.set_yticks([])

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        if len(analysis.get("series", [])) > 1:
            ax.legend(loc='upper left', framealpha=0.9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()

    def render_bar_chart(self, analysis: dict, title: str = "",
                         x_labels: list = None, y_limits: tuple = None) -> bytes:
        """Render a bar chart based on analysis."""
        plt.style.use(self.style)
        # Re-apply CJK font after style change
        if _CJK_FONT:
            plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(10, 5), dpi=self.fig_dpi)

        # Determine Y-axis scaling
        y_min, y_max = 0, 1
        if y_limits:
            y_min, y_max = y_limits

        y_range = y_max - y_min

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

        num_series = len(analysis.get("series", []))
        width = 0.8 / max(num_series, 1)

        for i, series in enumerate(analysis.get("series", [])):
            points = series.get("points", [])
            if not points:
                continue

            x_vals = np.array([p[0] for p in points])
            # Scale y-values
            y_vals = [y_min + p[1] * y_range for p in points]

            offset = (i - num_series / 2 + 0.5) * width
            color = colors[i % len(colors)]

            ax.bar(x_vals + offset, y_vals, width=width * 0.9,
                   color=color, label=series.get("name", f"Series {i+1}"),
                   alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(y_min if y_min < 0 else 0, y_max * 1.1)

        if x_labels:
            ax.set_xticks(np.linspace(0, 1, len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=9)

        if y_limits:
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.set_yticks([])

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        if num_series > 1:
            ax.legend(loc='upper right', framealpha=0.9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()

    def render(self, analysis: dict, title: str = "",
               x_labels: list = None, y_limits: tuple = None) -> bytes:
        """Render chart based on detected type."""
        chart_type = analysis.get("chart_type", "line")

        if chart_type == "bar":
            return self.render_bar_chart(analysis, title, x_labels, y_limits)
        else:
            return self.render_line_chart(analysis, title, x_labels, y_limits)


def analyze_and_render_chart(image_bytes: bytes, title: str = "",
                             x_labels: list = None, y_limits: tuple = None) -> bytes | None:
    """Main function to analyze a chart image and re-render it.

    Args:
        image_bytes: Raw image bytes of the chart
        title: Optional title for the new chart
        x_labels: Optional x-axis labels
        y_limits: Optional tuple of (y_min, y_max) for axis scaling

    Returns:
        PNG image bytes of the re-rendered chart, or None if failed
    """
    try:
        analyzer = ChartAnalyzer()
        analysis = analyzer.analyze(image_bytes)

        # Need at least one series with data
        if not analysis.get("series"):
            return None

        renderer = ChartRenderer()
        return renderer.render(analysis, title, x_labels, y_limits)

    except Exception as e:
        print(f"Chart rendering failed: {e}")
        return None
