import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

# Suppress minor matplotlib/seaborn warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")


def build_cohort_retention_matrix(conn):
    """
    TASK 1a-f: Builds the cohort retention matrix using advanced SQL CTEs.
    Returns the raw dataframe and the pivoted percentage matrix.
    """
    print("\n" + "=" * 50)
    print("TASK 1: COHORT RETENTION MATRIX")
    print("=" * 50)

    # FIXED SQL: Added f.customer_unique_id to the cohort_data CTE so it can be counted
    query = """
        WITH first_purchase AS (
            SELECT 
                customer_unique_id, 
                MIN(order_year_month) as cohort_month
            FROM fact_orders
            GROUP BY customer_unique_id
        ),
        customer_activity AS (
            SELECT 
                customer_unique_id, 
                order_year_month as activity_month
            FROM fact_orders
            GROUP BY customer_unique_id, order_year_month
        ),
        cohort_data AS (
            SELECT
                f.customer_unique_id,
                f.cohort_month,
                c.activity_month,
                CAST(SUBSTR(c.activity_month, 1, 4) AS INTEGER) - CAST(SUBSTR(f.cohort_month, 1, 4) AS INTEGER) as year_diff,
                CAST(SUBSTR(c.activity_month, 6, 2) AS INTEGER) - CAST(SUBSTR(f.cohort_month, 6, 2) AS INTEGER) as month_diff
            FROM first_purchase f
            JOIN customer_activity c ON f.customer_unique_id = c.customer_unique_id
        )
        SELECT
            cohort_month,
            (year_diff * 12) + month_diff AS period_number,
            COUNT(DISTINCT customer_unique_id) AS retained_users
        FROM cohort_data
        GROUP BY cohort_month, period_number
        ORDER BY cohort_month, period_number
    """

    df_raw = pd.read_sql_query(query, conn)

    # Pivot into matrix
    cohort_pivot = df_raw.pivot(index='cohort_month', columns='period_number', values='retained_users')

    # Period 0 is the cohort size. Divide everything by Period 0 to get percentages
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100

    print(f"Matrix calculated. Number of cohorts: {len(retention_matrix)}")
    return df_raw, retention_matrix, cohort_size


def plot_cohort_heatmap(retention_matrix, cohort_size):
    """
    TASK 1: Plots the professional seaborn heatmap of cohort retention.
    """
    # Filter out incomplete cohorts (less than 3 months of data)
    # A cohort has 3 months of data if period 3 is not NaN
    if 3 in retention_matrix.columns:
        valid_cohorts = retention_matrix[retention_matrix[3].notna()].copy()
    else:
        valid_cohorts = retention_matrix.copy()

    # Limit view to first 12 months for visual clarity
    cols_to_plot = [col for col in valid_cohorts.columns if col <= 12 and col > 0]
    plot_data = valid_cohorts[cols_to_plot]

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(plot_data, annot=True, fmt=".1f", cmap='Blues',
                vmin=0, vmax=plot_data.max().max(), cbar_kws={'label': 'Retention Rate (%)'}, ax=ax)

    # Text formatting fix to add % sign to annotations
    for t in ax.texts:
        if t.get_text() != "nan":
            t.set_text(t.get_text() + " %")
            t.set_fontsize(8)

    plt.title("Customer Retention by Acquisition Cohort (Olist E-Commerce)", fontsize=16, fontweight='bold', pad=20)
    plt.suptitle("Question: Which acquisition month cohorts have the BEST and WORST retention?", fontsize=12,
                 color='gray', y=0.96)
    plt.xlabel("Months Since First Purchase", fontweight='bold', labelpad=10)
    plt.ylabel("Acquisition Cohort (Month)", fontweight='bold', labelpad=10)

    # Identify stats for dict
    month_1_avg = valid_cohorts[1].mean()
    month_3_avg = valid_cohorts[3].mean() if 3 in valid_cohorts.columns else 0
    best_cohort_m3 = valid_cohorts[3].idxmax() if 3 in valid_cohorts.columns else valid_cohorts.index[0]
    worst_cohort_m3 = valid_cohorts[3].idxmin() if 3 in valid_cohorts.columns else valid_cohorts.index[0]

    # Calculate death month (steepest average drop)
    avg_retention = valid_cohorts.mean()
    drops = avg_retention.diff().abs()
    death_month = drops.idxmax() if len(drops) > 1 else 1

    # Annotate Best Cohort
    try:
        best_row_idx = list(plot_data.index).index(best_cohort_m3)
        col_3_idx = list(plot_data.columns).index(3)
        ax.annotate('Best M3 Retention', xy=(col_3_idx + 0.5, best_row_idx + 0.5),
                    xytext=(col_3_idx + 2, best_row_idx - 1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
                    fontsize=10, fontweight='bold', color='green', bbox=dict(facecolor='white', alpha=0.8))
    except ValueError:
        pass  # Handle case where period 3 doesn't exist

    plt.tight_layout()
    plt.savefig("cohort_retention_heatmap.png", dpi=300)
    plt.close()

    print(f"Heatmap generated. Best M3 Cohort: {best_cohort_m3}, Worst M3 Cohort: {worst_cohort_m3}")

    return {
        'best_cohort': best_cohort_m3,
        'worst_cohort': worst_cohort_m3,
        'avg_month1_retention': month_1_avg,
        'avg_month3_retention': month_3_avg,
        'death_month': death_month
    }


def plot_retention_curves_by_segment(conn):
    """
    TASK 2: Calculates and plots retention curves broken down by Region, Price, and Review Score.
    """
    print("\n" + "=" * 50)
    print("TASK 2: RETENTION CURVES BY SEGMENT")
    print("=" * 50)

    # Massive CTE to get segment info from the FIRST order, then track activity
    query = """
        WITH first_orders AS (
            SELECT 
                customer_unique_id, 
                order_id,
                order_year_month as cohort_month,
                customer_region,
                price_bucket,
                CASE 
                    WHEN review_score IN (1, 2) THEN '1-2 Stars'
                    WHEN review_score = 3 THEN '3 Stars'
                    WHEN review_score IN (4, 5) THEN '4-5 Stars'
                    ELSE 'Unrated'
                END as initial_satisfaction,
                ROW_NUMBER() OVER(PARTITION BY customer_unique_id ORDER BY order_purchase_timestamp ASC) as rn
            FROM fact_orders
        ),
        first_purchase_segments AS (
            SELECT * FROM first_orders WHERE rn = 1
        ),
        customer_activity AS (
            SELECT 
                customer_unique_id, 
                order_year_month as activity_month
            FROM fact_orders
            GROUP BY customer_unique_id, order_year_month
        ),
        segmented_cohort_data AS (
            SELECT
                f.customer_region,
                f.price_bucket,
                f.initial_satisfaction,
                f.cohort_month,
                (CAST(SUBSTR(c.activity_month, 1, 4) AS INTEGER) - CAST(SUBSTR(f.cohort_month, 1, 4) AS INTEGER)) * 12 + 
                (CAST(SUBSTR(c.activity_month, 6, 2) AS INTEGER) - CAST(SUBSTR(f.cohort_month, 6, 2) AS INTEGER)) as period_number
            FROM first_purchase_segments f
            JOIN customer_activity c ON f.customer_unique_id = c.customer_unique_id
        )
        SELECT 
            customer_region, price_bucket, initial_satisfaction, period_number,
            COUNT(*) as retained_users
        FROM segmented_cohort_data
        WHERE period_number <= 6
        GROUP BY customer_region, price_bucket, initial_satisfaction, period_number
    """
    df = pd.read_sql_query(query, conn)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    segments = [('customer_region', 'By Region'), ('price_bucket', 'By First Order Price'),
                ('initial_satisfaction', 'By First Review Score')]

    insight_high = 0
    insight_low = 0

    for i, (col, title) in enumerate(segments):
        ax = axes[i]

        # Calculate rates dynamically for the subplot
        pivot = df.groupby([col, 'period_number'])['retained_users'].sum().unstack(fill_value=0)
        rates = pivot.divide(pivot[0], axis=0) * 100

        for index, row in rates.iterrows():
            if index == 'Unknown' or index == 'Unrated': continue

            line, = ax.plot(rates.columns, row.values, marker='o', linewidth=2, label=index)

            # Annotate Period 1 and 3
            if 1 in rates.columns:
                ax.text(1, row[1] + 0.05, f"{row[1]:.1f}%", fontsize=9, color=line.get_color())
            if 3 in rates.columns:
                ax.text(3, row[3] + 0.05, f"{row[3]:.1f}%", fontsize=9, color=line.get_color())

            # Capture specific insight data for the print statement
            if col == 'initial_satisfaction':
                if index == '4-5 Stars' and 1 in rates.columns: insight_high = row[1]
                if index == '1-2 Stars' and 1 in rates.columns: insight_low = row[1]

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Months Since First Purchase")
        ax.set_ylabel("Retention Rate (%)")
        ax.set_xticks(range(7))
        ax.set_xlim(-0.5, 6.5)
        ax.legend(title=col.replace('_', ' ').title())

    plt.suptitle("Retention Decay Curves by Customer Segment (0-6 Months)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("retention_curves_by_segment.png", dpi=300)
    plt.close()

    print(f"Customers who rated their first order 4-5 stars retain at {insight_high:.2f}% after 1 month ")
    print(
        f"vs {insight_low:.2f}% for 1-2 star customers. That's a {(insight_high - insight_low):.2f} percentage point gap.")


def cohort_revenue_analysis(conn):
    """
    TASK 3: Calculates Cumulative Revenue per User and 6-Month LTV by Cohort.
    """
    print("\n" + "=" * 50)
    print("TASK 3: COHORT REVENUE & LTV ANALYSIS")
    print("=" * 50)

    query = """
        WITH first_purchase AS (
            SELECT 
                customer_unique_id, 
                MIN(order_year_month) as cohort_month
            FROM fact_orders
            GROUP BY customer_unique_id
        ),
        cohort_sizes AS (
            SELECT cohort_month, COUNT(DISTINCT customer_unique_id) as cohort_size
            FROM first_purchase
            GROUP BY cohort_month
        ),
        cohort_revenue AS (
            SELECT
                f.cohort_month,
                (CAST(SUBSTR(o.order_year_month, 1, 4) AS INTEGER) - CAST(SUBSTR(f.cohort_month, 1, 4) AS INTEGER)) * 12 + 
                (CAST(SUBSTR(o.order_year_month, 6, 2) AS INTEGER) - CAST(SUBSTR(f.cohort_month, 6, 2) AS INTEGER)) as period_number,
                SUM(o.payment_value) as period_revenue
            FROM first_purchase f
            JOIN fact_orders o ON f.customer_unique_id = o.customer_unique_id
            GROUP BY f.cohort_month, period_number
        )
        SELECT 
            r.cohort_month, 
            r.period_number, 
            r.period_revenue,
            s.cohort_size,
            r.period_revenue / s.cohort_size as avg_revenue_per_user
        FROM cohort_revenue r
        JOIN cohort_sizes s ON r.cohort_month = s.cohort_month
        WHERE r.period_number <= 6
        ORDER BY r.cohort_month, r.period_number
    """
    df = pd.read_sql_query(query, conn)

    # Calculate Cumulative Revenue Per User
    df['cumulative_revenue_per_user'] = df.groupby('cohort_month')['avg_revenue_per_user'].cumsum()

    # Extract 6-Month LTV (max cumulative revenue up to period 6)
    ltv_6m = df.groupby('cohort_month')['cumulative_revenue_per_user'].max().reset_index()
    ltv_6m.rename(columns={'cumulative_revenue_per_user': 'ltv_6m'}, inplace=True)

    # Assign Quarters for coloring
    def get_quarter(ym):
        month = int(ym[-2:])
        if month <= 3:
            return 'Q1'
        elif month <= 6:
            return 'Q2'
        elif month <= 9:
            return 'Q3'
        else:
            return 'Q4'

    ltv_6m['quarter'] = ltv_6m['cohort_month'].apply(get_quarter)

    # Plotting Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'Q1': '#185FA5', 'Q2': '#1D9E75', 'Q3': '#E67E22', 'Q4': '#E24B4A'}

    bars = ax.bar(ltv_6m['cohort_month'], ltv_6m['ltv_6m'],
                  color=ltv_6m['quarter'].map(colors))

    plt.title("6-Month Customer Lifetime Value (LTV) by Acquisition Cohort", fontsize=14, fontweight='bold')
    plt.suptitle("Question: Do customers acquired in different seasons hold different lifetime values?", fontsize=11,
                 color='gray')
    plt.xlabel("Acquisition Cohort")
    plt.ylabel("6-Month LTV (₹)")
    plt.xticks(rotation=45, ha='right')

    # Custom legend
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=q) for q, color in colors.items()]
    ax.legend(handles=legend_handles, title="Acquisition Quarter")

    # Add Payback threshold line
    ax.axhline(200, color='red', linestyle='--', alpha=0.7, label='CAC Threshold (₹200)')

    plt.tight_layout()
    plt.savefig("cohort_ltv_analysis.png", dpi=300)
    plt.close()

    best_ltv_row = ltv_6m.loc[ltv_6m['ltv_6m'].idxmax()]
    worst_ltv_row = ltv_6m.loc[ltv_6m['ltv_6m'].idxmin()]

    print(
        f"Best LTV cohort: {best_ltv_row['cohort_month']} with ₹{best_ltv_row['ltv_6m']:.2f} per customer over 6 months.")
    print(f"Worst LTV cohort: {worst_ltv_row['cohort_month']} with ₹{worst_ltv_row['ltv_6m']:.2f} per customer.")

    return ltv_6m['ltv_6m'].mean()


def generate_cohort_insights(conn, cohort_results):
    """
    TASK 4: Calculates strict D30/D90 retention, builds insights CSV, and prints executive summary.
    """
    print("\n" + "=" * 50)
    print("TASK 4: COHORT BUSINESS INSIGHT GENERATOR")
    print("=" * 50)

    # Calculate true D30 / D90 retention using julianday
    query = """
        WITH first_orders AS (
            SELECT customer_unique_id, MIN(order_purchase_timestamp) as first_date
            FROM fact_orders GROUP BY customer_unique_id
        ),
        subsequent_orders AS (
            SELECT 
                f.customer_unique_id, 
                julianday(o.order_purchase_timestamp) - julianday(f.first_date) as days_diff
            FROM first_orders f
            JOIN fact_orders o ON f.customer_unique_id = o.customer_unique_id
            WHERE o.order_purchase_timestamp > f.first_date
        )
        SELECT 
            (SELECT COUNT(*) FROM first_orders) as total_customers,
            (SELECT COUNT(DISTINCT customer_unique_id) FROM subsequent_orders WHERE days_diff <= 30) as d30_retained,
            (SELECT COUNT(DISTINCT customer_unique_id) FROM subsequent_orders WHERE days_diff <= 90) as d90_retained
    """
    df_retention = pd.read_sql_query(query, conn)

    total_cust = df_retention['total_customers'][0]
    d30_rate = (df_retention['d30_retained'][0] / total_cust) * 100 if total_cust > 0 else 0
    d90_rate = (df_retention['d90_retained'][0] / total_cust) * 100 if total_cust > 0 else 0

    avg_6m_ltv = cohort_results.get('avg_ltv', 150)  # Fallback to 150 if not passed
    revenue_lost = (1 - (d90_rate / 100)) * total_cust * avg_6m_ltv

    # Construct Insights Dataframe
    insights_data = [
        ['Overall D30 Retention', f"{d30_rate:.2f}%", "15.00%", "No" if d30_rate < 15 else "Yes",
         "Immediate post-purchase engagement is critically low."],
        ['Overall D90 Retention', f"{d90_rate:.2f}%", "8.00%", "No" if d90_rate < 8 else "Yes",
         "Long-term habit formation is failing; investigate product quality."],
        ['Revenue Lost to Churn', f"₹{revenue_lost:,.0f}", "N/A", "N/A",
         "Massive financial leak. Improving D90 by 2% yields millions."],
        ['Death Month', f"Month {cohort_results.get('death_month', 1)}", "Month 3", "N/A",
         "Deploy heavy win-back campaigns 2 weeks before this month."],
        ['Best Cohort (M3)', str(cohort_results.get('best_cohort', 'N/A')), "N/A", "N/A",
         "Reverse-engineer marketing campaigns from this month."]
    ]

    insights_df = pd.DataFrame(insights_data,
                               columns=['metric_name', 'value', 'benchmark', 'above_benchmark', 'business_implication'])
    insights_df.to_csv("cohort_insights.csv", index=False)

    print("\n" + "━" * 70)
    print(" 📈 COHORT ANALYSIS EXECUTIVE SUMMARY")
    print("━" * 70)
    for idx, row in insights_df.iterrows():
        print(f"➤ {row['metric_name']}: {row['value']} (Benchmark: {row['benchmark']})")
        print(f"   Implication: {row['business_implication']}\n")
    print("━" * 70)
    print("Output saved to 'cohort_insights.csv'. All charts generated successfully.")


if __name__ == "__main__":
    start_time = time.time()
    db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')

    try:
        conn = sqlite3.connect(db_path)

        # Task 1
        df_raw, retention_matrix, cohort_size = build_cohort_retention_matrix(conn)
        cohort_dict = plot_cohort_heatmap(retention_matrix, cohort_size)

        # Task 2
        plot_retention_curves_by_segment(conn)

        # Task 3
        avg_ltv = cohort_revenue_analysis(conn)
        cohort_dict['avg_ltv'] = avg_ltv

        # Task 4
        generate_cohort_insights(conn, cohort_dict)

        conn.close()
        elapsed = time.time() - start_time
        print(f"\n✅ Phase 3 Complete! Total Execution Time: {elapsed:.2f} seconds")

    except sqlite3.OperationalError as e:
        print(f"\n❌ DB Error: {e}")
        print("Please ensure 'ecommerce_analytics.db' exists in the current directory.")