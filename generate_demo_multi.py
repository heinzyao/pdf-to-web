import io

import fitz
import matplotlib.pyplot as plt


def create_bar_chart():
    """Create a bar chart using matplotlib."""
    plt.figure(figsize=(10, 6), dpi=150)

    products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    sales = [4500, 3200, 5100, 2800, 3900]

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    plt.bar(products, sales, color=colors)

    plt.title(
        "Product Sales Comparison Q4 2024", fontsize=16, fontweight="bold", pad=20
    )
    plt.ylabel("Sales Units", fontsize=12)
    plt.xlabel("Products", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def create_pie_chart():
    """Create a pie chart using matplotlib."""
    plt.figure(figsize=(8, 8), dpi=150)

    categories = ["Electronics", "Clothing", "Food", "Books", "Others"]
    values = [35, 25, 20, 10, 10]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    plt.pie(values, labels=categories, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.title(
        "Revenue Distribution by Category", fontsize=16, fontweight="bold", pad=20
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def create_multi_chart():
    """Create a multi-line chart using matplotlib."""
    plt.figure(figsize=(10, 6), dpi=150)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

    # Three product lines
    revenue_a = [1200, 1400, 1100, 1600, 1900, 2100]
    revenue_b = [800, 900, 1100, 1000, 1200, 1300]
    revenue_c = [1500, 1600, 1400, 1700, 1800, 1950]

    plt.plot(
        months, revenue_a, marker="o", linewidth=3, label="Product A", color="#2E86AB"
    )
    plt.plot(
        months, revenue_b, marker="s", linewidth=3, label="Product B", color="#A23B72"
    )
    plt.plot(
        months, revenue_c, marker="^", linewidth=3, label="Product C", color="#F18F01"
    )

    plt.title("Monthly Revenue Trends", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Revenue (USD)", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def create_demo_multi_chart_pdf(output_path="demo_multi.pdf"):
    """Create a PDF with multiple charts and text."""
    doc = fitz.open()

    # Page 1: Title and Introduction
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Comprehensive Business Report 2024",
        fontsize=24,
        fontname="helv",
        color=(0, 0, 0),
    )
    page.insert_text(
        (50, 80),
        "Quarterly Performance Analysis",
        fontsize=14,
        fontname="helv",
        color=(80, 80, 80),
    )
    page.insert_text(
        (50, 120),
        "This report provides a comprehensive overview of business performance",
        fontsize=12,
        fontname="helv",
    )
    page.insert_text(
        (50, 140),
        "across multiple product lines and categories.",
        fontsize=12,
        fontname="helv",
    )

    # Add Bar Chart
    bar_img = create_bar_chart()
    rect = fitz.Rect(50, 180, 550, 430)
    page.insert_image(rect, stream=bar_img)

    # Page 2: Pie Chart and Text
    page = doc.new_page()
    page.insert_text(
        (50, 50), "Revenue Distribution", fontsize=20, fontname="helv", color=(0, 0, 0)
    )
    page.insert_text(
        (50, 80),
        "Analysis of revenue sources across different categories.",
        fontsize=12,
        fontname="helv",
    )

    pie_img = create_pie_chart()
    rect = fitz.Rect(150, 120, 450, 420)
    page.insert_image(rect, stream=pie_img)

    # Page 3: Multi-line Chart
    page = doc.new_page()
    page.insert_text(
        (50, 50), "Monthly Trends", fontsize=20, fontname="helv", color=(0, 0, 0)
    )
    page.insert_text(
        (50, 80),
        "Comparing revenue growth across all product lines.",
        fontsize=12,
        fontname="helv",
    )

    multi_img = create_multi_chart()
    rect = fitz.Rect(50, 110, 550, 360)
    page.insert_image(rect, stream=multi_img)

    doc.save(output_path)
    print(f"Created {output_path}")


if __name__ == "__main__":
    create_demo_multi_chart_pdf()
