"""
E-Commerce Revenue Intelligence System - Phase 7: Final Delivery & Orchestration
Author: Vrushabh Gopal Gadhave
Description: Generates the Master Dashboard, Executive PDF Report, GitHub README,
and orchestrates the entire pipeline.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os
import subprocess
import time
import shutil
import warnings
from datetime import datetime
from typing import Dict, Any, List

# Suppress harmless matplotlib tight_layout warnings for gridspec tables
warnings.filterwarnings('ignore')

# Attempt to load ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️ ReportLab not installed. PDF generation will be skipped.")
    print("   To install: pip install reportlab")

# Set strict styling requirements
THEME = 'dark' # Options: 'dark' or 'light'
if THEME == 'dark':
    BG_COLOR = '#0D1117'
    CHART_BG = '#161B22'
    TEXT_COLOR = 'white'
    ACCENT = '#58A6FF'
else:
    BG_COLOR = '#FAFAFA'
    CHART_BG = 'white'
    TEXT_COLOR = '#111827'
    ACCENT = '#185FA5'

plt.rcParams['font.family'] = 'sans-serif' # Fallback for DejaVu Sans
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR

def format_inr(number: float) -> str:
    """Formats a number in Indian Rupee format (e.g., ₹1,50,000)"""
    try:
        is_negative = number < 0
        number = abs(int(number))
        s = str(number)
        if len(s) > 3:
            r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        else:
            r = s
        if number >= 10000000:
            return f"₹{number/10000000:.2f} Cr"
        return f"-₹{r}" if is_negative else f"₹{r}"
    except:
        return f"₹{number}"

def fetch_dashboard_kpis(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Fetches high-level metrics for the dashboard header."""
    kpis = {}

    # Revenue, Orders, AOV
    query = "SELECT SUM(payment_value) as rev, COUNT(DISTINCT order_id) as orders FROM fact_orders"
    res = pd.read_sql_query(query, conn)
    rev = res['rev'].iloc[0]
    orders = res['orders'].iloc[0]

    kpis['Total Revenue'] = format_inr(rev)
    kpis['Total Orders'] = f"{orders/1000:.1f}k"
    kpis['Avg Order Value'] = f"₹{rev/orders:,.0f}" if orders else "₹0"

    # Repeat Rate
    query_repeat = """
        SELECT 
            SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as repeat_rate 
        FROM (SELECT customer_unique_id, COUNT(DISTINCT order_id) as order_count FROM fact_orders GROUP BY customer_unique_id)
    """
    kpis['Repeat Customer Rate'] = f"{pd.read_sql_query(query_repeat, conn).iloc[0,0]:.1f}%"
    kpis['Overall Retention Rate'] = "1.78%" # Hardcoded from our Phase 3 deep dive for display

    return kpis

def build_master_dashboard(conn: sqlite3.Connection) -> None:
    """TASK 1: Creates a highly detailed 24x16 inch Master KPI Dashboard."""
    print("\n" + "="*50)
    print("TASK 1: MASTER KPI DASHBOARD")
    print("="*50)

    fig = plt.figure(figsize=(24, 16), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 3, 3], hspace=0.3, wspace=0.3)

    # ---------------- ROW 1: HEADER & KPIs ----------------
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(CHART_BG)
    ax_header.axis('off')

    ax_header.text(0.02, 0.7, "E-Commerce Revenue Leakage & Growth Intelligence System",
                   fontsize=24, fontweight='bold', color=TEXT_COLOR)
    ax_header.text(0.02, 0.4, f"Prepared by: Vrushabh Gopal Gadhave | Data Snapshot: 2016-2018 | Generated: {datetime.now().strftime('%d %b %Y')}",
                   fontsize=14, color=ACCENT)

    kpis = fetch_dashboard_kpis(conn)
    x_pos = 0.05
    for title, value in kpis.items():
        ax_header.text(x_pos, 0.1, title.upper(), fontsize=12, color='gray', fontweight='bold')
        ax_header.text(x_pos, -0.2, value, fontsize=28, color=TEXT_COLOR, fontweight='bold')
        x_pos += 0.20

    # ---------------- ROW 2: NATIVE CHARTS ----------------
    ax_trend = fig.add_subplot(gs[1, 0:2])
    ax_trend.set_facecolor(CHART_BG)
    trend_df = pd.read_sql_query("SELECT order_year_month, SUM(payment_value) as rev FROM fact_orders GROUP BY order_year_month", conn)
    trend_df = trend_df[trend_df['order_year_month'] >= '2017-01'].sort_values('order_year_month')
    ax_trend.plot(range(len(trend_df)), trend_df['rev'], color=ACCENT, linewidth=3, marker='o')
    ax_trend.set_xticks(range(len(trend_df)))
    ax_trend.set_xticklabels(trend_df['order_year_month'], rotation=45, ha='right')
    ax_trend.set_title("Monthly Revenue Trend (2017-2018)", color=TEXT_COLOR, fontweight='bold')
    ax_trend.grid(color='gray', alpha=0.2)

    ax_cat = fig.add_subplot(gs[1, 2])
    ax_cat.set_facecolor(CHART_BG)
    cat_df = pd.read_sql_query("SELECT product_category_name_english as cat, SUM(payment_value) as rev FROM fact_orders GROUP BY cat ORDER BY rev DESC LIMIT 5", conn)
    ax_cat.barh(cat_df['cat'][::-1], cat_df['rev'][::-1], color='#1D9E75')
    ax_cat.set_title("Top 5 Revenue Categories", color=TEXT_COLOR, fontweight='bold')
    ax_cat.grid(axis='x', color='gray', alpha=0.2)

    ax_donut = fig.add_subplot(gs[1, 3])
    ax_donut.set_facecolor(CHART_BG)
    try:
        rfm = pd.read_csv('rfm_segments.csv')
        counts = rfm['segment'].value_counts().head(5)
        ax_donut.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
                     colors=sns.color_palette("Set2"), textprops={'color': TEXT_COLOR})
        centre_circle = plt.Circle((0,0),0.70,fc=CHART_BG)
        ax_donut.add_artist(centre_circle)
        ax_donut.set_title("Top Customer Segments", color=TEXT_COLOR, fontweight='bold')
    except:
        ax_donut.text(0.5, 0.5, "RFM Data Missing", ha='center', color=TEXT_COLOR)

    # ---------------- ROW 3: IMAGE INGESTION ----------------
    def load_image_to_ax(ax, filepath, title):
        ax.set_facecolor(CHART_BG)
        ax.axis('off')
        ax.set_title(title, color=TEXT_COLOR, fontweight='bold')
        if os.path.exists(filepath):
            img = mpimg.imread(filepath)
            ax.imshow(img, aspect='auto')
        else:
            ax.text(0.5, 0.5, f"Missing:\n{filepath}", ha='center', va='center', color='red')

    ax_cohort = fig.add_subplot(gs[2, 0])
    load_image_to_ax(ax_cohort, 'cohort_retention_heatmap.png', "Cohort Retention Matrix")

    ax_funnel = fig.add_subplot(gs[2, 1])
    load_image_to_ax(ax_funnel, 'conversion_funnel.png', "E-Commerce Funnel Conversion")

    ax_geo = fig.add_subplot(gs[2, 2])
    load_image_to_ax(ax_geo, 'geo_yoy_growth.png', "Geographic YoY Growth")

    ax_table = fig.add_subplot(gs[2, 3])
    ax_table.set_facecolor(CHART_BG)
    ax_table.axis('off')
    ax_table.set_title("Q3 2018 Revenue Forecast Scenarios", color=TEXT_COLOR, fontweight='bold')

    table_data = [
        ["Month", "Conservative (-15%)", "Base Forecast", "Aggressive (+15%)"],
        ["Jul '18", "₹14.7 L", "₹17.3 L", "₹19.9 L"],
        ["Aug '18", "₹17.6 L", "₹20.7 L", "₹23.8 L"],
        ["Sep '18", "₹17.3 L", "₹20.3 L", "₹23.3 L"]
    ]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color=TEXT_COLOR)
        cell.set_facecolor('#1F2937' if row == 0 else CHART_BG)

    fig.text(0.5, 0.02, "Data: Olist Brazilian E-Commerce Dataset | B.Tech AIML Portfolio Project | 2026",
             ha='center', fontsize=12, color='gray')

    plt.tight_layout()
    plt.savefig('master_dashboard.png', dpi=400, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Master Dashboard saved successfully.")

def generate_executive_report() -> None:
    """TASK 2: Generates the Executive PDF Report."""
    print("\n" + "="*50)
    print("TASK 2: AUTOMATED EXECUTIVE REPORT (PDF)")
    print("="*50)

    if not HAS_REPORTLAB: return

    doc = SimpleDocTemplate("executive_report.pdf", pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#0D1117'), spaceAfter=20)
    subtitle_style = ParagraphStyle('SubStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#185FA5'), spaceAfter=30)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#0D1117'), spaceBefore=20, spaceAfter=10)
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.spaceAfter = 10

    elements = []

    # PAGE 1: COVER
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("E-Commerce Revenue Intelligence Report", title_style))
    elements.append(Paragraph("Revenue Leakage Analysis & Growth Recommendations", subtitle_style))
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("<b>Date:</b> June 2026", normal_style))
    elements.append(Paragraph("<b>Prepared by:</b> Vrushabh Gopal Gadhave (Data Analyst)", normal_style))
    elements.append(Paragraph("<b>Target Audience:</b> Leadership & Strategy Teams", normal_style))
    elements.append(PageBreak())

    # PAGE 2: EXECUTIVE SUMMARY
    elements.append(Paragraph("Executive Summary", title_style))
    elements.append(Paragraph("<b>Situation:</b> The platform experienced steady volume growth in late 2017, however, profit margins and customer retention have demonstrated significant volatility. Management requires an analytical deep dive to identify revenue leakage bottlenecks and prescribe data-driven interventions.", normal_style))

    elements.append(Paragraph("<b>Key Findings:</b>", h2_style))
    elements.append(Paragraph("• <b>Revenue Leakage:</b> ₹75.7 Lakhs in revenue is actively at risk due to high churn correlated with late delivery times (>10 days).", normal_style))
    elements.append(Paragraph("• <b>Cohort Decay:</b> D30 retention is critically low at 1.25%, indicating a failure in immediate post-purchase engagement.", normal_style))
    elements.append(Paragraph("• <b>Value Concentration:</b> The RFM model reveals that the top 25% 'Platinum' tier customers drive an outsized 66.0% of total historical revenue.", normal_style))

    elements.append(Paragraph("<b>Top 3 Recommendations:</b>", h2_style))
    elements.append(Paragraph("1. Re-negotiate SLAs with 3PL carriers handling routes in the underperforming North region.", normal_style))
    elements.append(Paragraph("2. Deploy a targeted win-back discount campaign specifically to the 11,111 customers identified in the 'At Risk' segment.", normal_style))
    elements.append(Paragraph("3. Optimize top-of-funnel checkout flows to mitigate the primary drop-off between Order Placed and Order Approved.", normal_style))
    elements.append(PageBreak())

    def add_image(elements, filename, w, h):
        if os.path.exists(filename):
            elements.append(RLImage(filename, width=w, height=h))
            elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph(f"<i>[Image {filename} not found]</i>", normal_style))

    # PAGES 3-5
    elements.append(Paragraph("Revenue Root Cause Analysis", title_style))
    add_image(elements, 'rca_waterfall.png', 500, 300)
    elements.append(Paragraph("The waterfall decomposition isolates the core drivers of month-over-month revenue variance. Volume effects heavily dominate price effects, indicating our growth relies on acquiring new users rather than expanding basket sizes.", normal_style))
    add_image(elements, 'pareto_analysis.png', 500, 300)
    elements.append(PageBreak())

    elements.append(Paragraph("Customer Cohort & RFM Segmentation", title_style))
    add_image(elements, 'rfm_segment_distribution.png', 500, 300)
    elements.append(Paragraph("RFM (Recency, Frequency, Monetary) segmentation classifies the entire user base into actionable groups. We have successfully isolated 14,908 'Champions' ready for VIP upselling.", normal_style))
    add_image(elements, 'cohort_retention_heatmap.png', 500, 300)
    elements.append(PageBreak())

    elements.append(Paragraph("Funnel Performance & Revenue Forecast", title_style))
    add_image(elements, 'conversion_funnel.png', 500, 300)
    add_image(elements, 'model_comparison.png', 500, 300)
    elements.append(PageBreak())

    # PAGE 6: Appendix / Strategy Table
    elements.append(Paragraph("Actionable Strategy Matrix", title_style))
    data = [
        ['Business Area', 'Finding', 'Action Required', 'Priority'],
        ['Logistics', '>10 day delivery causes churn', 'Renegotiate 3PL SLAs', 'High'],
        ['Retention', 'D30 retention at 1.25%', 'Deploy Day-14 email sequences', 'High'],
        ['Catalog', 'Bottom 55 cats = 20% rev', 'Delist low-margin SKUs', 'Medium'],
        ['CRM', '11k customers At Risk', 'Trigger 15% win-back discount', 'High'],
        ['Forecasting', 'Auto category growing', 'Shift ad-spend to Auto segment', 'Medium']
    ]
    t = Table(data, colWidths=[1.2*inch, 2*inch, 2.5*inch, 1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#185FA5')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F3F4F6')),
        ('GRID', (0,0), (-1,-1), 1, colors.white)
    ]))
    elements.append(t)

    doc.build(elements)
    print("Executive Report PDF saved successfully.")

def generate_github_readme() -> None:
    """TASK 3: Generates the comprehensive GitHub README.md using safe string concatenation."""
    print("\n" + "="*50)
    print("TASK 3: GITHUB README GENERATOR")
    print("="*50)

    readme_lines = [
        "# 📊 E-Commerce Revenue Leakage & Growth Intelligence System",
        "![Python](https://img.shields.io/badge/Python-3.12-blue)",
        "![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey)",
        "![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-green)",
        "![Prophet](https://img.shields.io/badge/Prophet-Forecasting-orange)",
        "![ReportLab](https://img.shields.io/badge/ReportLab-PDF_Automation-red)",
        "",
        "## 📌 Problem Statement",
        "The e-commerce platform experienced 12% YoY revenue growth, yet profit margins declined and retention remained volatile. Management needed to know exactly **where revenue was leaking** and **which levers to pull** to recover it. This end-to-end system ingests raw transaction data, builds a star schema, and applies advanced analytics (RFM, Cohorts, Funnels, ML Forecasting) to prescribe actionable business strategies.",
        "",
        "## 💾 Dataset",
        "| Attribute | Detail |",
        "| :--- | :--- |",
        "| **Source** | Olist Brazilian E-Commerce (Kaggle) |",
        "| **Scale** | 100,000+ Orders, 90,000+ Customers |",
        "| **Timeframe** | Sept 2016 - Oct 2018 |",
        "| **Schema** | 9 interconnected raw CSV tables |",
        "",
        "## 🏗️ Architecture Pipeline",
        "```text",
        "Raw CSVs ",
        "   ↳ Data Engineering (Pandas + SQLite Star Schema)",
        "        ↳ Phase 2: Root Cause Analysis (Waterfall, Pareto)",
        "        ↳ Phase 3: Cohort & LTV Analysis (SQL CTEs)",
        "        ↳ Phase 4: RFM Segmentation Engine (Quantiles)",
        "        ↳ Phase 5: Conversion Funnel Analysis",
        "        ↳ Phase 6: Time Series Forecasting (Prophet, SARIMA)",
        "             ↳ Phase 7: Delivery (Matplotlib Dashboard & PDF Automation)",
        "```",
        "",
        "## 💡 Key Business Findings",
        "* 🚨 **Revenue at Risk:** Identified **₹75.7 Lakhs** bleeding due to elevated churn correlated strictly with >10-day delivery SLA breaches.",
        "* 🏆 **Value Concentration:** The top 25% of customers (Platinum Tier) drive **66.0%** of total historical revenue.",
        "* 📉 **Cohort Decay:** Discovered a massive drop-off at Month 1 (D30 retention at 1.25%), prompting a pivot toward immediate post-purchase engagement.",
        "* 📦 **Catalog Bloat:** Pareto analysis proved that 55 product categories (76% of the catalog) generate barely 20% of revenue.",
        "",
        "## 🛠️ Tech Stack",
        "| Tool | Purpose |",
        "| :--- | :--- |",
        "| **Python** | Core logic, scripting, orchestration |",
        "| **SQLite3** | Analytical data warehouse, Star Schema |",
        "| **Pandas / NumPy** | Data wrangling, feature engineering |",
        "| **Matplotlib / Seaborn** | Statistical visualizations, Dashboards |",
        "| **Prophet / pmdarima** | Time series forecasting (Holt-Winters fallback) |",
        "| **ReportLab** | Automated PDF executive reporting |",
        "",
        "## 🚀 How to Run Locally",
        "1. Clone the repository.",
        "2. Install dependencies: `pip install -r requirements.txt`",
        "3. Download the Olist dataset from Kaggle and place the 9 CSVs in the `/data` folder.",
        "4. Run the master orchestrator: `python run_pipeline.py`",
        "5. Check the `/outputs` folder for your PDF, Dashboard, and CSVs!",
        "",
        "## 💼 Business Impact",
        "By transitioning from descriptive reporting to prescriptive modeling, this system allows stakeholders to isolate exact bottlenecks. If the recommendations generated by the Funnel Optimization matrix are implemented to increase D90 retention by just 2%, it yields an estimated **₹1.79 Cr** in recovered Lifetime Value.",
        "",
        "## 🎯 Interview Questions This Project Prepares Me For:",
        "1. *How do you handle missing or dirty data in a 100k+ row dataset?* (Handled via Phase 1 ETL pipeline).",
        "2. *Can you write complex SQL?* (Demonstrated via recursive CTEs in Phase 3 Cohorts).",
        "3. *How do you segment customers?* (Demonstrated via programmatic NTILE RFM scoring in Phase 4).",
        "4. *How do you predict future trends?* (Demonstrated via multi-model Prophet/SARIMA testing in Phase 6).",
        "5. *How do you present to stakeholders?* (Demonstrated via the automated PDF and Dashboard in Phase 7).",
        "",
        "---",
        "**Prepared by:** Vrushabh Gopal Gadhave  ",
        "**Role:** Data Analyst | B.Tech AIML"
    ]

    # Using utf-8 encoding to support emojis
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))
    print("README.md generated successfully.")

def generate_project_files() -> None:
    """TASK 4: Generates requirements, gitignore, and orchestrator script."""
    print("\n" + "="*50)
    print("TASK 4: SYSTEM FILES GENERATION")
    print("="*50)

    reqs = [
        "pandas==2.2.0", "numpy==1.26.0", "matplotlib==3.8.0", "seaborn==0.13.0",
        "scikit-learn==1.4.0", "scipy==1.12.0", "statsmodels==0.14.1",
        "prophet==1.1.5", "pmdarima==2.0.4", "reportlab==4.1.0"
    ]
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reqs))

    ignores = ["data/", "__pycache__/", "*.pyc", "ecommerce_analytics.db", "venv/", ".env", "outputs/"]
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write("\n".join(ignores))

    script_lines = [
        "import os, subprocess, time",
        "",
        "phases = [",
        "    ('Phase 1: ETL & Schema', 'main.py'),",
        "    ('Phase 2: Root Cause Analysis', 'phase2_rca.py'),",
        "    ('Phase 3: Cohorts & LTV', 'phase3_cohorts.py'),",
        "    ('Phase 4: RFM Segmentation', 'phase4_rfm.py'),",
        "    ('Phase 5: Funnel Analysis', 'phase5_funnel.py'),",
        "    ('Phase 6: ML Forecasting', 'phase6_forecast.py'),",
        "    ('Phase 7: Dashboard & Delivery', 'Phase7_Delivery.py')",
        "]",
        "",
        "print('🚀 Starting E-Commerce Analytics Pipeline...')",
        "start = time.time()",
        "",
        "for name, script in phases:",
        "    print(f'\\n➤ Running {name}...')",
        "    result = subprocess.run(['python', script])",
        "    if result.returncode != 0:",
        "        print(f'❌ Error in {script}. Pipeline halted.')",
        "        break",
        "    print(f'✅ {name} Completed.')",
        "",
        "print(f'\\n🎉 Pipeline finished in {time.time() - start:.2f} seconds!')"
    ]
    # Fixed the encoding issue here!
    with open("run_pipeline.py", "w", encoding="utf-8") as f:
        f.write("\n".join(script_lines))
    print("System files created.")

def generate_linkedin_post() -> None:
    """TASK 5: Generates 3 highly detailed LinkedIn post variants using safe string concatenation."""
    print("\n" + "="*50)
    print("TASK 5: LINKEDIN POST GENERATOR")
    print("="*50)

    post_lines = [
        "--------------------------------------------------",
        "VARIANT 1: TECHNICAL ACHIEVEMENT (For Data/Tech Audience)",
        "--------------------------------------------------",
        "Just completed my most comprehensive data project yet! 🚀 \n",
        "I built an end-to-end E-Commerce Revenue Intelligence System using Python and SQLite, processing over 100,000 orders. I didn't just want to build dashboards; I wanted to drive business value. \n",
        "Through advanced SQL CTEs and programmatic RFM segmentation, I identified ₹75.7 Lakhs in revenue actively at risk due to logistics SLA breaches. I then modeled 6-month Customer Lifetime Value (CLV) and built a multi-model time series forecasting engine (comparing SARIMA, Prophet, and Linear Regression).\n",
        "To wrap it up, I automated the entire delivery pipeline to generate an Executive PDF Report and a Matplotlib KPI Dashboard.\n",
        "Check out the full repository and architecture here: [GitHub Link]",
        "I am actively looking for Data Analyst roles where I can bring this level of end-to-end analytical rigor to your team! #DataAnalytics #Python #SQL #MachineLearning #DataScience #DataEngineering\n",
        "",
        "--------------------------------------------------",
        "VARIANT 2: STORY TONE (For Broader Audience)",
        "--------------------------------------------------",
        "\"Revenue is up 12%, but why are profit margins shrinking?\" 🤔\n",
        "This is the exact business problem I set out to solve in my latest portfolio project. Using a dataset of 100k+ e-commerce orders, I dug past the surface-level metrics to find the real story.\n",
        "The culprit? A massive retention leak. \n",
        "By building a Cohort Retention Matrix, I discovered that 98% of customers were churning by Month 1. Digging deeper into a Root Cause Analysis (RCA), the data revealed that late deliveries (>10 days) were devastating customer satisfaction and bleeding potential lifetime value. \n",
        "Data is only valuable if it drives action. So, I built an automated system that segments customers and prescribes exact marketing strategies based on their behavior.\n",
        "If you love data-driven storytelling as much as I do, check out the full project breakdown here: [GitHub Link] #Analytics #BusinessIntelligence #DataStorytelling\n",
        "",
        "--------------------------------------------------",
        "VARIANT 3: HIRING FOCUSED TONE (Direct & Punchy)",
        "--------------------------------------------------",
        "If your team is hiring Data Analysts who can bridge the gap between raw data and executive strategy, let's connect. 🤝\n",
        "I recently built an automated E-Commerce Intelligence System that features:",
        "✅ Data Engineering: Raw CSVs to SQLite Star Schema",
        "✅ Advanced SQL: Complex CTEs for Cohort & Retention modeling",
        "✅ Customer Intel: NTILE-based RFM Segmentation & LTV prediction",
        "✅ Machine Learning: Revenue forecasting via SARIMA & Exponential Smoothing",
        "✅ Automation: Python script that auto-generates multi-page Executive PDFs\n",
        "I'm a B.Tech AIML graduate ready to bring this technical depth and business acumen to a forward-thinking analytics team. \n",
        "Full code and methodology: [GitHub Link] #Hiring #DataAnalyst #DataAnalytics #OpenToWork"
    ]

    with open("linkedin_posts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(post_lines))
    print("LinkedIn post variants saved to 'linkedin_posts.txt'.")

def run_complete_pipeline() -> None:
    """FINAL TASK: Organizes outputs and concludes the project."""
    print("\n" + "="*50)
    print("FINAL TASK: PIPELINE ORCHESTRATION")
    print("="*50)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    for file in os.listdir():
        if file in ['ecommerce_analytics.db', 'requirements.txt', 'README.md', 'run_pipeline.py']:
            continue
        if file.endswith('.png') or file.endswith('.pdf') or file.endswith('.csv') or file.endswith('.txt'):
            try:
                shutil.move(file, os.path.join("outputs", file))
            except shutil.Error:
                os.replace(file, os.path.join("outputs", file))

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" ✅ PROJECT PIPELINE COMPLETE")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" 📊 Dashboard:     outputs/master_dashboard.png")
    print(" 📄 Report:        outputs/executive_report.pdf")
    print(" 🗄️  Database:      ecommerce_analytics.db")
    print(" 📁 All outputs:   /outputs directory")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" Ready to present to recruiters. Good luck! 🚀")

if __name__ == "__main__":
    db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')
    try:
        conn = sqlite3.connect(db_path)
        build_master_dashboard(conn)
        generate_executive_report()
        generate_github_readme()
        generate_project_files()
        generate_linkedin_post()
        run_complete_pipeline()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"\n❌ DB Error: {e}")

# --- END OF FILE ---