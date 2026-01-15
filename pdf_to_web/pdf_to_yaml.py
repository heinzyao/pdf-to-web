"""PDF to YAML extraction module.

This module extracts structured content from PDF files and converts them to YAML format,
with intelligent image filtering and chart recognition.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import yaml
from PIL import Image


class ImageAnalyzer:
    """Analyzes and filters images from PDF to identify decorative vs content images."""

    def __init__(self):
        self.image_hashes = Counter()  # Track duplicate images by hash
        self.image_sizes = Counter()  # Track image sizes
        self.all_images = []  # Store all image info for analysis

    def compute_image_hash(self, image_bytes: bytes) -> str:
        """Compute a hash to identify duplicate images."""
        return hashlib.md5(image_bytes).hexdigest()

    def register_image(
        self, image_bytes: bytes, width: int, height: int, page_idx: int
    ) -> dict:
        """Register an image for later analysis."""
        img_hash = self.compute_image_hash(image_bytes)
        size_key = f"{width}x{height}"

        self.image_hashes[img_hash] += 1
        self.image_sizes[size_key] += 1

        info = {
            "hash": img_hash,
            "size_key": size_key,
            "width": width,
            "height": height,
            "page_idx": page_idx,
            "bytes": image_bytes,
        }
        self.all_images.append(info)
        return info

    def analyze_patterns(self, total_pages: int) -> tuple[set, set]:
        """Analyze image patterns to identify decorative images.

        Returns:
            Tuple of (decorative_hashes, decorative_sizes)
        """
        # Images appearing on >40% of pages are likely decorative (logos, backgrounds)
        threshold = max(3, total_pages * 0.4)

        decorative_hashes = {
            h for h, count in self.image_hashes.items() if count >= threshold
        }

        decorative_sizes = {
            s for s, count in self.image_sizes.items() if count >= threshold
        }

        return decorative_hashes, decorative_sizes

    def is_decorative(
        self, img_info: dict, decorative_hashes: set, decorative_sizes: set
    ) -> bool:
        """Check if an image is decorative based on patterns."""
        # Check by hash (exact duplicate)
        if img_info["hash"] in decorative_hashes:
            return True

        # Check by size pattern (same dimensions appearing many times)
        if img_info["size_key"] in decorative_sizes:
            return True

        # Too small (likely icon)
        if img_info["width"] < 150 or img_info["height"] < 150:
            return True

        return False

    def is_likely_chart(self, img_info: dict, image_bytes: bytes) -> bool:
        """Heuristically determine if an image is likely a chart."""
        try:
            from io import BytesIO

            img = Image.open(BytesIO(image_bytes))

            width, height = img.size
            aspect_ratio = width / height if height > 0 else 1

            # Charts are typically wider than tall (landscape)
            if aspect_ratio < 1.2:
                return False

            # Large landscape images are likely charts
            if width > 1000 and height > 400:
                return True

            return False
        except Exception:
            return False


def _extract_text_blocks(page: fitz.Page) -> list[dict]:
    """Extract text blocks from a PDF page with position info."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    text_blocks = []

    for block in blocks:
        if block["type"] == 0:  # Text block
            lines_text = []
            max_font_size = 0
            is_bold = False

            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    font_size = span.get("size", 0)
                    if font_size > max_font_size:
                        max_font_size = font_size
                    font_name = span.get("font", "").lower()
                    if "bold" in font_name or "heavy" in font_name:
                        is_bold = True

                if line_text.strip():
                    lines_text.append(line_text.strip())

            full_text = "\n".join(lines_text)
            if full_text.strip():
                text_blocks.append(
                    {
                        "text": full_text.strip(),
                        "bbox": block["bbox"],
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                    }
                )

    return text_blocks


def _classify_text_blocks(
    text_blocks: list[dict], page_height: float
) -> tuple[str, list[dict]]:
    """Classify text blocks into title and content based on font size and position."""
    if not text_blocks:
        return "", []

    sorted_blocks = sorted(text_blocks, key=lambda b: b["bbox"][1])
    max_font_size = max(b["font_size"] for b in text_blocks) if text_blocks else 0

    title = ""
    content = []

    for block in sorted_blocks:
        text = block["text"]
        font_size = block["font_size"]
        is_bold = block["is_bold"]
        y_pos = block["bbox"][1]

        # Title detection
        is_title_candidate = (
            (font_size >= max_font_size * 0.9 and font_size > 12)
            or (is_bold and font_size > 14)
            or (y_pos < page_height * 0.15 and font_size > 12)
        )

        if not title and is_title_candidate and len(text) < 200:
            title = text.replace("\n", " ").strip()
        else:
            content.append({"type": "text", "value": text})

    return title, content


def _extract_tables(page_plumber) -> list[dict]:
    """Extract tables from a PDF page using pdfplumber."""
    tables = []

    try:
        extracted_tables = page_plumber.extract_tables()

        for table in extracted_tables:
            if not table or not table[0]:
                continue

            headers = table[0] if table else []
            rows = table[1:] if len(table) > 1 else []

            clean_headers = [str(cell).strip() if cell else "" for cell in headers]
            clean_rows = [
                [str(cell).strip() if cell else "" for cell in row] for row in rows
            ]

            if any(clean_headers) or any(any(row) for row in clean_rows):
                tables.append(
                    {
                        "type": "table",
                        "headers": clean_headers,
                        "rows": clean_rows,
                    }
                )

    except Exception as e:
        print(f"Warning: Failed to extract tables: {e}")

    return tables


def _detect_highlighted_page(text_blocks: list[dict]) -> bool:
    """Detect if a page is highlighted based on text styling."""
    if not text_blocks:
        return False

    bold_count = sum(1 for b in text_blocks if b["is_bold"])
    large_font_count = sum(1 for b in text_blocks if b["font_size"] > 16)

    total_blocks = len(text_blocks)
    if total_blocks == 0:
        return False

    return (bold_count / total_blocks > 0.3) or (large_font_count / total_blocks > 0.2)


def pdf_to_yaml(pdf_path: str, yaml_output_dir: str) -> str:
    """Convert a PDF file to YAML format with intelligent image processing.

    Args:
        pdf_path: Path to the input PDF file.
        yaml_output_dir: Directory to save the YAML file and media.

    Returns:
        Path to the created YAML file.
    """
    print("Starting pdf_to_yaml conversion...")
    pdf_file = Path(pdf_path)
    yaml_dir = Path(yaml_output_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing PDF: {pdf_file}, output dir: {yaml_dir}")

    media_dir = yaml_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    print("Opening PDF with fitz...")
    doc = fitz.open(str(pdf_file))
    print(f"PDF opened, pages: {len(doc)}")
    pdf_plumber = pdfplumber.open(str(pdf_file))
    print("PDF opened with pdfplumber")

    total_pages = len(doc)
    analyzer = ImageAnalyzer()

    # First pass: collect all images for pattern analysis
    print("Analyzing image patterns...")
    page_images = {}  # page_idx -> list of image info

    for page_idx in range(total_pages):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        page_images[page_idx] = []

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                info = analyzer.register_image(image_bytes, width, height, page_idx)
                info["xref"] = xref
                info["img_idx"] = img_idx
                info["ext"] = base_image["ext"]
                page_images[page_idx].append(info)
            except Exception:
                pass

    # Analyze patterns to identify decorative images
    decorative_hashes, decorative_sizes = analyzer.analyze_patterns(total_pages)
    print(f"Found {len(decorative_hashes)} decorative image patterns (by hash)")
    print(f"Found {len(decorative_sizes)} decorative size patterns")

    # Second pass: extract content with filtered images
    pages_data = []
    highlighted_sections = []
    chart_counter = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_plumber_obj = pdf_plumber.pages[page_idx]

        page_rect = page.rect
        page_height = page_rect.height

        # Extract text
        text_blocks = _extract_text_blocks(page)
        title, content = _classify_text_blocks(text_blocks, page_height)

        # Extract tables
        tables = _extract_tables(page_plumber_obj)

        # Process images with filtering
        media = []
        for img_info in page_images.get(page_idx, []):
            # Skip decorative images
            if analyzer.is_decorative(img_info, decorative_hashes, decorative_sizes):
                continue

            image_bytes = img_info["bytes"]
            width = img_info["width"]
            height = img_info["height"]
            ext = img_info["ext"]
            img_idx = img_info["img_idx"]

            # Check if this is likely a chart
            is_chart = analyzer.is_likely_chart(img_info, image_bytes)

            # Save the image
            filename = f"page_{page_idx + 1}_content_{img_idx + 1}.{ext}"
            filepath = media_dir / filename

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            # Convert to PNG if needed
            if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
                try:
                    with Image.open(filepath) as img:
                        png_filename = f"page_{page_idx + 1}_content_{img_idx + 1}.png"
                        png_filepath = media_dir / png_filename
                        img.save(png_filepath, "PNG")
                        filepath.unlink()
                        filename = png_filename
                except Exception:
                    pass

            aspect_ratio = width / height if height > 0 else 1.0

            if is_chart:
                chart_counter += 1
                chart_data_ai = None
                rendered_bytes = None

                # First, try AI-powered chart analysis (most accurate)
                try:
                    from pdf_to_web.ai_chart_analyzer import (
                        analyze_chart_with_ai,
                        render_chart_from_ai_data,
                    )

                    print(f"  Analyzing chart with AI...")
                    chart_data_ai = analyze_chart_with_ai(image_bytes)

                    if chart_data_ai:
                        rendered_bytes = render_chart_from_ai_data(chart_data_ai)
                        if rendered_bytes:
                            print(f"  AI chart extraction successful!")
                except Exception as e:
                    print(f"  AI chart analysis unavailable: {e}")

                # Fallback: Try OpenCV + matplotlib approach
                if not rendered_bytes:
                    try:
                        from pdf_to_web.chart_renderer import analyze_and_render_chart

                        # Get data from OCR if available
                        x_labels = None
                        y_limits = None
                        try:
                            from pdf_to_web.chart_extractor import ChartExtractor

                            extractor = ChartExtractor()
                            ocr_text = extractor.extract_text(image_bytes)
                            chart_data = extractor.parse_chart_data(ocr_text)
                            if chart_data:
                                x_labels = chart_data.get("x_axis")
                                y_limits = chart_data.get("y_axis_range")
                        except Exception:
                            pass

                        rendered_bytes = analyze_and_render_chart(
                            image_bytes,
                            title=title or f"Chart {chart_counter}",
                            x_labels=x_labels,
                            y_limits=y_limits,
                        )
                        if rendered_bytes:
                            print(f"  Fallback chart rendering successful.")
                    except Exception as e:
                        print(f"  Fallback chart rendering failed: {e}")

                if rendered_bytes:
                    # Save the re-rendered chart
                    rendered_filename = (
                        f"page_{page_idx + 1}_chart_{img_idx + 1}_rendered.png"
                    )
                    rendered_path = media_dir / rendered_filename
                    with open(rendered_path, "wb") as f:
                        f.write(rendered_bytes)

                    # Get dimensions of rendered image
                    with Image.open(rendered_path) as rimg:
                        rwidth, rheight = rimg.size
                        raspect = rwidth / rheight if rheight > 0 else 1.0

                    # If AI extracted data, create interactive ECharts chart
                    if chart_data_ai and chart_data_ai.get("series"):
                        chart_id = f"chart_{page_idx + 1}_{img_idx + 1}"
                        x_axis = chart_data_ai.get("x_axis", {})
                        y_axis = chart_data_ai.get("y_axis", {})
                        series_list = chart_data_ai.get("series", [])

                        # Convert to ECharts format
                        echarts_series = []
                        for s in series_list:
                            chart_type = chart_data_ai.get("chart_type", "line")
                            echarts_series.append(
                                {
                                    "name": s.get("name", "Series"),
                                    "type": chart_type
                                    if chart_type in ["line", "bar"]
                                    else "line",
                                    "data": s.get("data", []),
                                    "smooth": True,
                                }
                            )

                        media_item = {
                            "type": "chart",
                            "chart_id": chart_id,
                            "title": chart_data_ai.get("title") or title,
                            "categories": x_axis.get("values", []),
                            "y_min": y_axis.get("min", 0),
                            "y_max": y_axis.get("max"),
                            "series": echarts_series,
                            # Keep image as fallback
                            "fallback_path": f"media/{rendered_filename}",
                        }
                        print(f"  Interactive ECharts chart created: {chart_id}")
                    else:
                        # No AI data, use image with chart styling
                        media_item = {
                            "type": "chart_image",
                            "path": f"media/{rendered_filename}",
                            "width": rwidth,
                            "height": rheight,
                            "aspect_ratio": raspect,
                            "is_chart": True,
                        }

                    media.append(media_item)
                    print(f"  Chart saved: {rendered_filename}")
                else:
                    # Fallback to original image
                    media.append(
                        {
                            "type": "image",
                            "path": f"media/{filename}",
                            "width": width,
                            "height": height,
                            "aspect_ratio": aspect_ratio,
                            "is_chart": True,
                        }
                    )
                    print(f"  Chart kept as original: {filename}")
            else:
                media.append(
                    {
                        "type": "image",
                        "path": f"media/{filename}",
                        "width": width,
                        "height": height,
                        "aspect_ratio": aspect_ratio,
                    }
                )

        # Add tables to media
        media.extend(tables)

        is_highlighted = _detect_highlighted_page(text_blocks)

        page_data = {
            "slide_number": page_idx + 1,
            "title": title,
            "content": content,
            "media": media,
            "is_highlighted": is_highlighted,
            "layout": f"PDF Page {page_idx + 1}",
        }

        if title or content or media:
            pages_data.append(page_data)
            if is_highlighted:
                highlighted_sections.append(
                    {
                        "slide_number": page_idx + 1,
                        "title": title,
                        "content": content[:3],
                    }
                )

    doc.close()
    pdf_plumber.close()

    # Summary
    total_images = len(analyzer.all_images)
    filtered_images = sum(
        1
        for img in analyzer.all_images
        if analyzer.is_decorative(img, decorative_hashes, decorative_sizes)
    )
    print(f"\nImage filtering summary:")
    print(f"  Total images extracted: {total_images}")
    print(f"  Decorative images filtered: {filtered_images}")
    print(f"  Content images kept: {total_images - filtered_images}")
    print(f"  Chart-like images identified: {chart_counter}")

    # Build output
    cover_title = pdf_file.stem
    if pages_data and pages_data[0].get("title"):
        cover_title = pages_data[0]["title"].replace("\n", " ").strip()

    output_data = {
        "title": pdf_file.stem,
        "cover_title": cover_title,
        "hero_image": None,
        "slides": pages_data,
        "highlighted_sections": highlighted_sections,
        "total_slides": len(pages_data),
    }

    yaml_path = yaml_dir / f"{pdf_file.stem}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            output_data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    return str(yaml_path)
