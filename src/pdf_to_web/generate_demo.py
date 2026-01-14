
import fitz
import io
import matplotlib.pyplot as plt
import numpy as np

def create_demo_chart():
    """Create a sample chart image using matplotlib."""
    plt.figure(figsize=(10, 6), dpi=150)
    
    years = [2020, 2021, 2022, 2023, 2024]
    revenue = [1200, 1500, 1800, 2200, 2600]
    
    plt.plot(years, revenue, marker='o', linewidth=3, color='#2E86AB')
    plt.title('Annual Revenue Growth', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Revenue (USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Remove top/right spines for cleaner look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return buf.getvalue()

def create_demo_pdf(output_path="demo.pdf"):
    """Create a PDF with text and the sample chart."""
    doc = fitz.open()
    page = doc.new_page()
    
    # Add Title
    page.insert_text((50, 50), "Financial Report 2024", fontsize=24, fontname="helv", color=(0, 0, 0))
    page.insert_text((50, 80), "This is a demo PDF generated to test the chart extraction capabilities.", fontsize=12)
    
    # Add Chart Image
    img_bytes = create_demo_chart()
    
    # Define image rectangle (x0, y0, x1, y1)
    rect = fitz.Rect(50, 150, 550, 450)
    page.insert_image(rect, stream=img_bytes)
    
    # Add Mock OCR Text (Invisible text for testing OCR logic if needed, 
    # but our tool uses PyMuPDF extract_image then Tesseract. 
    # So we strictly need the image to contain the text visually.)
    
    doc.save(output_path)
    print(f"Created {output_path}")

if __name__ == "__main__":
    create_demo_pdf()
