"""AI-powered chart and infographic analysis using Google Gemini Vision API.

This module uses the new google-genai SDK for intelligent data extraction
from chart and infographic images.
"""

import base64
import json
import os
import re
from io import BytesIO

from PIL import Image

# Lazy import for google.genai
_client = None


def _get_client():
    """Get or create Gemini client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Get one from https://aistudio.google.com/app/apikey"
            )
        try:
            from google import genai
            _client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )
    return _client


# Enhanced prompt for both charts AND infographics
CHART_EXTRACTION_PROMPT = """Analyze this image carefully. It may be a chart, graph, infographic, or data visualization.

Your task is to extract ALL numerical data and text visible in the image.

Return a JSON object with this format:
{
    "image_type": "bar_chart" | "line_chart" | "pie_chart" | "scatter_plot" | "area_chart" | "infographic" | "data_table" | "mixed" | "unknown",
    "title": "Title if visible, or null",
    "x_axis": {
        "label": "X-axis label if visible, or null",
        "values": ["value1", "value2", ...]
    },
    "y_axis": {
        "label": "Y-axis label if visible, or null",
        "min": 0,
        "max": 100
    },
    "series": [
        {
            "name": "Series name or category",
            "data": [10, 20, 30, ...],
            "color": "#hexcolor if identifiable"
        }
    ],
    "data_points": [
        {"label": "Category A", "value": 100},
        {"label": "Category B", "value": 200}
    ],
    "text_content": ["Any important text visible in the image"],
    "notes": "Additional observations"
}

CRITICAL INSTRUCTIONS:
1. Extract ALL visible numbers, even if approximate
2. For infographics with icons/statistics, use data_points array
3. For traditional charts, use series array with proper x_axis values
4. Read axis labels and tick marks carefully
5. If multiple data series exist, identify each separately
6. For pie charts, estimate percentages from visual proportions
7. Return ONLY valid JSON, no markdown formatting
8. If you cannot extract meaningful data, set image_type to "unknown"
"""


def analyze_chart_with_ai(image_bytes: bytes) -> dict | None:
    """Analyze a chart/infographic image using Gemini Vision API.

    Args:
        image_bytes: Raw bytes of the image.

    Returns:
        Structured data dict, or None if analysis failed.
    """
    try:
        client = _get_client()
    except (ImportError, ValueError) as e:
        print(f"AI chart analysis unavailable: {e}")
        return None

    try:
        # Prepare the image
        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to bytes
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_data = img_buffer.getvalue()

        # Use the new Client API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": CHART_EXTRACTION_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(img_data).decode("utf-8")
                            }
                        }
                    ]
                }
            ],
            config={
                "temperature": 0.1,
                "max_output_tokens": 4096,
            }
        )

        # Parse response
        response_text = response.text.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        chart_data = json.loads(response_text)

        if not isinstance(chart_data, dict):
            print("AI response is not a valid JSON object")
            return None

        image_type = chart_data.get("image_type", "unknown")
        if image_type == "unknown":
            print("AI could not identify image type")
            return None

        # Normalize to chart_type for compatibility
        chart_type_map = {
            "bar_chart": "bar",
            "line_chart": "line",
            "pie_chart": "pie",
            "area_chart": "line",
            "scatter_plot": "scatter",
            "infographic": "bar",  # Render infographics as bar charts
            "data_table": "bar",
            "mixed": "bar",
        }
        chart_data["chart_type"] = chart_type_map.get(image_type, "bar")

        # Handle infographic data_points -> convert to series format
        if chart_data.get("data_points") and not chart_data.get("series"):
            data_points = chart_data["data_points"]
            if data_points:
                labels = [dp.get("label", f"Item {i+1}") for i, dp in enumerate(data_points)]
                values = [dp.get("value", 0) for dp in data_points]
                
                chart_data["x_axis"] = chart_data.get("x_axis", {})
                chart_data["x_axis"]["values"] = labels
                chart_data["series"] = [{
                    "name": "Value",
                    "data": values
                }]
                
                # Set y_axis range
                if values:
                    chart_data["y_axis"] = chart_data.get("y_axis", {})
                    chart_data["y_axis"]["min"] = 0
                    chart_data["y_axis"]["max"] = max(values) * 1.1

        # Check if we have usable data
        has_series = bool(chart_data.get("series"))
        has_data_points = bool(chart_data.get("data_points"))
        
        if not has_series and not has_data_points:
            print("AI could not extract chart data")
            return None

        series_count = len(chart_data.get("series", []))
        print(f"  AI extracted: {image_type} with {series_count} series")

        return chart_data

    except json.JSONDecodeError as e:
        print(f"Failed to parse AI response as JSON: {e}")
        return None
    except Exception as e:
        print(f"AI chart analysis failed: {e}")
        return None


def render_chart_from_ai_data(chart_data: dict, output_path: str = None) -> bytes | None:
    """Render a chart from AI-extracted data using matplotlib.

    Args:
        chart_data: Structured chart data from AI analysis.
        output_path: Optional path to save the rendered chart.

    Returns:
        PNG image bytes, or None if rendering failed.
    """
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np

    # Setup Chinese font
    chinese_fonts = [
        'Noto Sans CJK TC', 'Noto Sans TC', 'Heiti TC', 'PingFang TC',
        'PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS',
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            break

    try:
        chart_type = chart_data.get("chart_type", "bar")
        title = chart_data.get("title", "")
        x_axis = chart_data.get("x_axis", {})
        y_axis = chart_data.get("y_axis", {})
        series_list = chart_data.get("series", [])
        
        # Professional color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', 
                  '#5C946E', '#7B68EE', '#FF6B6B', '#4ECDC4', '#45B7D1']

        if not series_list:
            return None

        x_values = x_axis.get("values", [])
        if not x_values and series_list:
            x_values = [str(i+1) for i in range(len(series_list[0].get("data", [])))]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

        if chart_type == "bar":
            x_pos = np.arange(len(x_values))
            num_series = len(series_list)
            width = 0.8 / max(num_series, 1)

            for i, series in enumerate(series_list):
                data = series.get("data", [])
                # Ensure data matches x_values length
                if len(data) < len(x_values):
                    data = data + [0] * (len(x_values) - len(data))
                elif len(data) > len(x_values):
                    data = data[:len(x_values)]
                
                offset = (i - num_series / 2 + 0.5) * width
                color = series.get("color") or colors[i % len(colors)]
                
                bars = ax.bar(x_pos + offset, data, width=width, 
                             label=series.get("name", f"Series {i+1}"),
                             color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
                
                # Add value labels on bars
                for bar, val in zip(bars, data):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{val:.0f}' if isinstance(val, (int, float)) else str(val),
                               ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_values, rotation=45 if len(x_values) > 6 else 0, 
                              ha='right' if len(x_values) > 6 else 'center', fontsize=10)

        elif chart_type == "line":
            for i, series in enumerate(series_list):
                data = series.get("data", [])
                color = series.get("color") or colors[i % len(colors)]
                ax.plot(range(len(data)), data, marker='o', linewidth=2.5,
                        label=series.get("name", f"Series {i+1}"), color=color, markersize=6)

            if x_values:
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(x_values, rotation=45 if len(x_values) > 6 else 0, 
                                  ha='right' if len(x_values) > 6 else 'center', fontsize=10)

        elif chart_type == "pie":
            if series_list:
                data = series_list[0].get("data", [])
                labels = x_values if x_values else [f"Segment {i+1}" for i in range(len(data))]
                # Filter out zero values
                non_zero = [(l, d) for l, d in zip(labels, data) if d > 0]
                if non_zero:
                    labels, data = zip(*non_zero)
                    ax.pie(data, labels=labels, colors=colors[:len(data)], 
                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax.axis('equal')

        # Styling
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        if x_axis.get("label"):
            ax.set_xlabel(x_axis["label"], fontsize=12, labelpad=10)

        if y_axis.get("label"):
            ax.set_ylabel(y_axis["label"], fontsize=12, labelpad=10)

        if y_axis.get("min") is not None and y_axis.get("max") is not None:
            y_max = y_axis["max"]
            if isinstance(y_max, (int, float)):
                ax.set_ylim(y_axis.get("min", 0), y_max * 1.15)

        if len(series_list) > 1 and chart_type != "pie":
            ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=150)
        plt.close(fig)
        buf.seek(0)
        result = buf.getvalue()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(result)

        return result

    except Exception as e:
        print(f"Chart rendering from AI data failed: {e}")
        import traceback
        traceback.print_exc()
        return None
