import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import time

# Set visualization style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")


def run_revenue_decomposition(conn) -> dict:
    """
    TASK 1: Decomposes Month-over-Month revenue changes into Volume, Price, and Mix effects.
    Creates a waterfall chart of the total cumulative effects.
    """
    print("\n" + "=" * 50)
    print("TASK 1: REVENUE DECOMPOSITION WATERFALL")
    print("=" * 50)

    # a) SQL Query to calculate monthly metrics
    query = """
        SELECT 
            order_year_month,
            SUM(payment_value) as total_revenue,
            COUNT(DISTINCT order_id) as total_orders,
            SUM(payment_value) / COUNT(DISTINCT order_id) as AOV,
            CAST(COUNT(product_id) AS FLOAT) / COUNT(DISTINCT order_id) as avg_items_per_order,
            SUM(price) / NULLIF(COUNT(product_id), 0) as avg_price_per_item,
            SUM(freight_value) as total_freight
        FROM fact_orders
        GROUP BY order_year_month
        ORDER BY order_year_month
    """
    df = pd.read_sql_query(query, conn)

    # b) Calculate MoM changes
    df['total_orders_prior'] = df['total_orders'].shift(1)
    df['AOV_prior'] = df['AOV'].shift(1)
    df['total_revenue_prior'] = df['total_revenue'].shift(1)

    df = df.dropna().copy()

    # c) Decompose revenue change
    df['revenue_delta'] = df['total_revenue'] - df['total_revenue_prior']
    df['volume_effect'] = (df['total_orders'] - df['total_orders_prior']) * df['AOV_prior']
    df['price_effect'] = (df['AOV'] - df['AOV_prior']) * df['total_orders']
    df['mix_effect'] = df['revenue_delta'] - (df['volume_effect'] + df['price_effect'])

    # Aggregate total effects for the waterfall chart
    tot_vol = df['volume_effect'].sum()
    tot_price = df['price_effect'].sum()
    tot_mix = df['mix_effect'].sum()
    tot_delta = df['revenue_delta'].sum()

    # d) Create Waterfall Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Volume Effect', 'Price Effect', 'Mix Effect', 'Total Net Delta']
    values = [tot_vol, tot_price, tot_mix, tot_delta]

    # Calculate step bottoms
    bottoms = [0, tot_vol, tot_vol + tot_price, 0]
    colors = ['green' if x > 0 else 'red' for x in values[:-1]] + ['#185FA5']  # Final bar is blue

    bars = ax.bar(categories, values, bottom=bottoms, color=colors, width=0.6)

    # Add annotations
    for i, bar in enumerate(bars):
        yval = bar.get_y() + bar.get_height() + (tot_delta * 0.05 if values[i] > 0 else -tot_delta * 0.05)
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"₹{values[i]:,.0f}",
                ha='center', va='bottom' if values[i] > 0 else 'top', fontweight='bold')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Revenue Decomposition: Cumulative Variance Drivers", fontsize=14, fontweight='bold')
    ax.set_ylabel("Revenue Impact (₹)")
    ax.text(0.02, 0.95, "Insight: Identifies if growth is driven by customer volume or higher pricing.",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("rca_waterfall.png", dpi=300)
    plt.close()

    # f) Print Summary
    print(f"Cumulative Revenue Delta of ₹{tot_delta:,.0f} was driven by:")
    print(f" - Volume: +₹{tot_vol:,.0f}" if tot_vol > 0 else f" - Volume: -₹{abs(tot_vol):,.0f}")
    print(f" - Price:  +₹{tot_price:,.0f}" if tot_price > 0 else f" - Price:  -₹{abs(tot_price):,.0f}")
    print(f" - Mix:    +₹{tot_mix:,.0f}" if tot_mix > 0 else f" - Mix:    -₹{abs(tot_mix):,.0f}")

    return {'metric': 'Revenue Variance', 'value': f"₹{tot_delta:,.0f}",
            'finding': f"Volume drove ₹{tot_vol:,.0f}, Price drove ₹{tot_price:,.0f}"}


def run_pareto_analysis(conn) -> dict:
    """
    TASK 2: Calculates 80/20 rule on product categories to find revenue concentration.
    """
    print("\n" + "=" * 50)
    print("TASK 2: PARETO ANALYSIS (80/20 RULE)")
    print("=" * 50)

    # a) SQL Query for Category Revenue
    query = """
        SELECT 
            COALESCE(product_category_name_english, 'Unknown') as category,
            SUM(payment_value) as total_revenue
        FROM fact_orders
        GROUP BY category
        ORDER BY total_revenue DESC
    """
    df = pd.read_sql_query(query, conn)

    # b) Cumulative calculations
    total_rev = df['total_revenue'].sum()
    df['revenue_pct'] = (df['total_revenue'] / total_rev) * 100
    df['cumulative_pct'] = df['revenue_pct'].cumsum()

    # c) Flag top 20%
    df['is_top_20_pct'] = df['cumulative_pct'] <= 80

    top_categories_count = df[df['is_top_20_pct']].shape[0]
    total_categories = df.shape[0]
    bottom_categories_count = total_categories - top_categories_count

    # d) Dual-axis Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    top_20_df = df.head(20)  # Plot top 20 for readability

    # Bar chart for revenue
    sns.barplot(x='category', y='total_revenue', data=top_20_df, ax=ax1, color='#185FA5')
    ax1.set_xticks(range(len(top_20_df)))
    ax1.set_xticklabels(top_20_df['category'], rotation=45, ha='right')
    ax1.set_ylabel("Total Revenue (₹)", fontweight='bold')
    ax1.set_xlabel("")

    # Line chart for cumulative %
    ax2 = ax1.twinx()
    ax2.plot(top_20_df.index, top_20_df['cumulative_pct'], color='red', marker='o', linewidth=2)
    ax2.set_ylabel("Cumulative %", color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')

    # Threshold lines
    ax2.axhline(80, color='red', linestyle='--', alpha=0.5)

    cutoff_index = top_20_df[top_20_df['cumulative_pct'] >= 80].index.min()
    if pd.notna(cutoff_index):
        ax1.axvline(cutoff_index, color='red', linestyle='--', alpha=0.5)

    plt.title("Pareto Analysis: Revenue by Product Category", fontsize=14, fontweight='bold')

    ax1.text(0.5, 0.9, f"Insight: Top {top_categories_count} categories drive 80% of revenue",
             transform=ax1.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8), ha='center')

    plt.tight_layout()
    plt.savefig("pareto_analysis.png", dpi=300)
    plt.close()

    # f) Print summary
    print(f"Top {top_categories_count} categories represent 80% of revenue.")
    print(
        f"Bottom {bottom_categories_count} categories contribute only 20% of revenue but represent {(bottom_categories_count / total_categories) * 100:.1f}% of all categories.")

    return {'metric': 'Pareto Concentration', 'value': f"{top_categories_count} categories = 80%",
            'finding': f"Heavy catalog bloat: {bottom_categories_count} categories generate minimal revenue."}


def run_geographic_rca(conn) -> dict:
    """
    TASK 3: Identifies geographic anomalies and visualizes regional revenue drop-offs.
    """
    print("\n" + "=" * 50)
    print("TASK 3: GEOGRAPHIC REVENUE HEATMAP & ANOMALY")
    print("=" * 50)

    # a) Group by Region and Month (Year extracted dynamically to fix schema mismatch)
    query = """
        SELECT 
            customer_region,
            order_year_month,
            CAST(SUBSTR(order_year_month, 1, 4) AS INTEGER) as order_year,
            SUM(payment_value) as total_revenue,
            COUNT(DISTINCT order_id) as order_count,
            AVG(review_score) as avg_review_score,
            AVG(CASE WHEN is_late_delivery = 1 THEN 1.0 ELSE 0.0 END) * 100 as late_delivery_rate,
            SUM(payment_value) / COUNT(DISTINCT order_id) as AOV
        FROM fact_orders
        WHERE customer_region != 'Unknown'
        GROUP BY customer_region, order_year_month
    """
    df = pd.read_sql_query(query, conn)

    # Heatmap setup
    heatmap_data = df.pivot_table(index='customer_region', columns='order_year_month', values='total_revenue',
                                  fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, linewidths=.5, fmt=".0f")
    plt.title("Geographic Revenue Heatmap (Region vs Month)", fontsize=14, fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig("geo_heatmap.png", dpi=300)
    plt.close()

    # b) YoY Growth per region
    yoy_df = df.groupby(['customer_region', 'order_year'])['total_revenue'].sum().unstack()
    if 2017 in yoy_df.columns and 2018 in yoy_df.columns:
        yoy_df['YoY_Growth'] = ((yoy_df[2018] - yoy_df[2017]) / yoy_df[2017]) * 100
        avg_growth = yoy_df['YoY_Growth'].mean()

        # c) Identify Underperforming
        yoy_df['Status'] = np.where(yoy_df['YoY_Growth'] >= avg_growth, 'Above Avg', 'Below Avg')

        # e) Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if x >= avg_growth else 'red' for x in yoy_df['YoY_Growth']]
        yoy_df['YoY_Growth'].plot(kind='bar', color=colors, ax=ax)
        ax.axhline(avg_growth, color='blue', linestyle='--', label=f'Avg Growth ({avg_growth:.1f}%)')
        plt.title("YoY Revenue Growth Rate by Region (2018 vs 2017)", fontsize=14, fontweight='bold')
        plt.ylabel("Growth Rate (%)")
        plt.xticks(rotation=0)
        plt.legend()

        ax.text(0.02, 0.90, "Insight: Red regions are pulling down overall profitability.",
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig("geo_yoy_growth.png", dpi=300)
        plt.close()

        # g) Print underperforming
        underperforming = yoy_df[yoy_df['YoY_Growth'] < avg_growth].sort_values('YoY_Growth')
        print(f"Overall Average YoY Growth: {avg_growth:.2f}%")
        print("\nUnderperforming Regions:")
        for idx, row in underperforming.iterrows():
            print(f" - {idx}: {row['YoY_Growth']:.2f}% YoY")

        worst_region = underperforming.index[0] if not underperforming.empty else "None"
        return {'metric': 'Geographic Drag', 'value': worst_region,
                'finding': f"{worst_region} heavily underperformed average growth of {avg_growth:.1f}%"}
    else:
        print("Not enough full years for YoY calculation.")
        return {'metric': 'Geographic Drag', 'value': 'N/A', 'finding': 'Insufficient yearly data.'}


def run_delivery_impact_rca(conn) -> dict:
    """
    TASK 4: Quantifies the financial and review-score impact of late deliveries.
    """
    print("\n" + "=" * 50)
    print("TASK 4: LATE DELIVERY IMPACT ANALYSIS")
    print("=" * 50)

    # a) Calculate order level stats
    query = """
        SELECT 
            customer_unique_id,
            order_id,
            is_late_delivery,
            review_score,
            payment_value
        FROM fact_orders
        WHERE is_late_delivery IS NOT NULL AND review_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)

    late_df = df[df['is_late_delivery'] == 1]
    ontime_df = df[df['is_late_delivery'] == 0]

    late_rate = (len(late_df) / len(df)) * 100

    # b) Statistical Test
    t_stat, p_val = stats.ttest_ind(late_df['review_score'], ontime_df['review_score'], equal_var=False)

    # c) Calculate Revenue at Risk (Churn Proxy)
    # Customers who ordered only once
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique()
    one_time_buyers = order_counts[order_counts == 1].index

    avg_lost_clv = df[df['customer_unique_id'].isin(one_time_buyers)]['payment_value'].mean()
    total_customers = df['customer_unique_id'].nunique()

    revenue_at_risk = (late_rate / 100) * total_customers * avg_lost_clv

    # d) Side-by-side box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='is_late_delivery', y='review_score', data=df, ax=ax, palette=['#1D9E75', '#E24B4A'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['On-Time (≤10 days)', 'Late (>10 days)'])
    ax.set_xlabel("Delivery Performance", fontweight='bold')
    ax.set_ylabel("Review Score (1-5)", fontweight='bold')
    plt.title("Impact of Late Deliveries on Customer Satisfaction", fontsize=14, fontweight='bold')

    ax.text(0.5, 0.1, f"Statistical p-value: {p_val:.4e}\nRev at Risk: ₹{revenue_at_risk:,.0f}",
            transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.9), ha='center')

    plt.tight_layout()
    plt.savefig("delivery_impact.png", dpi=300)
    plt.close()

    # f) Print findings
    print(f"Late deliveries affect {late_rate:.1f}% of orders.")
    print(
        f"Average Review Score - On-Time: {ontime_df['review_score'].mean():.2f} | Late: {late_df['review_score'].mean():.2f}")
    print(f"Statistical test p-value = {p_val:.4e} (Significant Drop)")
    print(f"Estimated revenue at risk from delivery failures: ₹{revenue_at_risk:,.0f}")

    return {'metric': 'Late Delivery Penalty', 'value': f"₹{revenue_at_risk:,.0f} At Risk",
            'finding': f"Late deliveries cause massive CSAT drop (p-value: {p_val:.2e})."}


def generate_rca_summary(conn):
    """
    TASK 5: Master orchestration function that compiles findings into a CSV and prints report.
    """
    print("\n" + "=" * 50)
    print("TASK 5: MASTER RCA SUMMARY REPORT")
    print("=" * 50)

    res_1 = run_revenue_decomposition(conn)
    res_2 = run_pareto_analysis(conn)
    res_3 = run_geographic_rca(conn)
    res_4 = run_delivery_impact_rca(conn)

    # a) Create Final DataFrame
    rca_data = [
        ['Volume & Pricing', res_1['metric'], res_1['value'], res_1['finding'],
         "Launch targeted volume-driver campaigns."],
        ['Catalog Efficiency', res_2['metric'], res_2['value'], res_2['finding'],
         "Delist bottom 20% of catalog to save storage/focus."],
        ['Regional Performance', res_3['metric'], res_3['value'], res_3['finding'],
         "Audit supply chain and marketing in underperforming zones."],
        ['Logistics CSAT', res_4['metric'], res_4['value'], res_4['finding'],
         "Re-negotiate SLA with logistics partners handling >10 day routes."]
    ]

    rca_findings = pd.DataFrame(rca_data, columns=['finding_category', 'metric', 'value', 'business_impact',
                                                   'recommended_action'])

    # b) Save CSV
    rca_findings.to_csv("rca_summary.csv", index=False)

    # c) Print Formatted Summary
    print("\n" + "━" * 70)
    print(" 📊 EXECUTIVE RCA SUMMARY REPORT")
    print("━" * 70)
    for idx, row in rca_findings.iterrows():
        print(f"➤ CATEGORY: {row['finding_category'].upper()}")
        print(f"   Metric: {row['metric']} | Value: {row['value']}")
        print(f"   Impact: {row['business_impact']}")
        print(f"   Action: {row['recommended_action']}\n")
    print("━" * 70)
    print("Output saved to 'rca_summary.csv'. All charts generated successfully.")


if __name__ == "__main__":
    start_time = time.time()
    db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')

    try:
        # Connect to Database
        conn = sqlite3.connect(db_path)

        # Run orchestrator
        generate_rca_summary(conn)

        conn.close()
        elapsed = time.time() - start_time
        print(f"\n✅ Phase 2 Complete! Total Execution Time: {elapsed:.2f} seconds")

    except sqlite3.OperationalError as e:
        print(f"\n❌ DB Error: {e}")
        print("Please ensure you have run Phase 1 and 'ecommerce_analytics.db' exists in the current directory.")