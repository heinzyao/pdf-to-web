from pdf_to_web.chart_extractor import ChartExtractor
from pdf_to_web.chart_renderer import ChartRenderer


def test_extractor_logic():
    print("Testing Extractor Logic...")
    extractor = ChartExtractor()

    # Simulate OCR text containing years and numbers
    mock_ocr = """
    Market Growth
    2,500
    2,000
    1,500
    1,000
    500
    0
    2020 2021 2022 2023 2024
    Revenue (USD)
    """

    data = extractor.parse_chart_data(mock_ocr)
    print("Parsed Data:", data)

    assert data["chart_type"] == "line"
    assert "2020" in data["x_axis"]
    assert data["y_axis_range"] == [0, 2500.0]
    print("Extractor Logic: PASS")


def test_renderer_logic():
    print("\nTesting Renderer Logic...")
    renderer = ChartRenderer()

    # Simulate analysis result (normalized coordinates 0-1)
    analysis = {
        "chart_type": "line",
        "series": [
            {
                "name": "Revenue",
                "points": [
                    (0.0, 0.2),
                    (0.5, 0.5),
                    (1.0, 0.8),
                ],  # Normalized Y: 0.2 -> 0.8
            }
        ],
    }

    # Real limits: 0 to 1000
    y_limits = (0, 1000)
    x_labels = ["2020", "2021", "2022"]

    # Render
    img_bytes = renderer.render(
        analysis, title="Test Chart", x_labels=x_labels, y_limits=y_limits
    )

    print(f"Generated Image Bytes: {len(img_bytes)}")
    assert len(img_bytes) > 0
    print("Renderer Logic: PASS")


if __name__ == "__main__":
    test_extractor_logic()
    test_renderer_logic()
