# Changelog

## [0.1.0] - 2026-01-14

### Added
- Initial release of pdf-to-web converter
- PDF content extraction with PyMuPDF and pdfplumber
- Smart image filtering to remove decorative images
- AI-powered chart analysis using Google Gemini Vision API
- Interactive ECharts rendering for extracted chart data
- Magazine-style HTML template with responsive design
- Full CJK/Chinese language support
- CLI interface with `convert`, `build`, and `run` commands

### Technical Details
- Uses `google-genai` SDK (v1.57+) for Gemini 2.0 Flash
- Fallback to OpenCV + OCR chart extraction when AI unavailable
- Automatic Y-axis range detection from extracted data
- Two-column magazine layout with drop caps and professional typography
