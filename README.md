# pdf-to-web

Convert PDF documents to professional, magazine-style web pages with AI-powered chart extraction.

## Features

- **Intelligent Content Extraction**: Extracts text, tables, and images from PDFs using PyMuPDF and pdfplumber
- **Smart Image Filtering**: Automatically filters decorative images (logos, backgrounds, icons) keeping only content-relevant visuals
- **AI Chart Recognition**: Uses Google Gemini Vision API to analyze charts and infographics, extracting structured data
- **Interactive Charts**: Converts static chart images to interactive ECharts visualizations with tooltips and zoom
- **Magazine-Style Layout**: Generates professional HTML with two-column layouts, drop caps, and responsive design
- **Chinese/CJK Support**: Full support for Traditional Chinese and other CJK languages

## Installation

```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install package
uv pip install -e .
```

## Configuration

For AI-powered chart extraction, set your Google API key:

```bash
export GOOGLE_API_KEY='your-api-key'
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

```bash
# One-step conversion (recommended)
pdf-to-web run input.pdf -o ./output

# Step-by-step conversion
pdf-to-web convert input.pdf -o ./yaml_output  # PDF to YAML
pdf-to-web build data.yaml -o ./html_output    # YAML to HTML
```

## Output

The tool generates:
- `output/{filename}.yaml` - Structured content in YAML format
- `output/{filename}.html` - Professional magazine-style web page
- `output/media/` - Extracted and re-rendered images/charts

## Architecture

```
src/pdf_to_web/
├── cli.py              # Command-line interface
├── pdf_to_yaml.py      # PDF content extraction
├── yaml_to_html.py     # HTML generation with Jinja2
├── ai_chart_analyzer.py # Gemini Vision chart analysis
├── chart_extractor.py  # OCR-based chart extraction (fallback)
├── chart_renderer.py   # Matplotlib chart rendering
└── templates/
    └── index.html      # Magazine-style HTML template
```

## Dependencies

- PyMuPDF (fitz) - PDF parsing
- pdfplumber - Table extraction
- google-genai - Gemini Vision API
- Jinja2 - HTML templating
- Matplotlib - Chart rendering
- ECharts (CDN) - Interactive charts in HTML

## License

MIT
