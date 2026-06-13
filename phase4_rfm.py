import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from datetime import datetime

# Suppress minor warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")


def calculate_rfm_scores(conn, snapshot_date='2018-10-01'):
    """
    TASK 1: Calculates Recency, Frequency, Monetary metrics and applies NTILE(5) scoring.
    """
    print("\n" + "=" * 50)
    print("TASK 1: RFM SCORE CALCULATION")
    print("=" * 50)

    snapshot = pd.to_datetime(snapshot_date)

    # a) Calculate raw RFM metrics
    query = """
        SELECT 
            customer_unique_id,
            MAX(order_purchase_timestamp) as last_order_date,
            COUNT(DISTINCT order_id) as frequency,
            SUM(payment_value) as monetary
        FROM fact_orders
        GROUP BY customer_unique_id
    """
    df_rfm = pd.read_sql_query(query, conn)

    # Convert dates and calculate recency
    df_rfm['last_order_date'] = pd.to_datetime(df_rfm['last_order_date'])
    df_rfm['recency'] = (snapshot - df_rfm['last_order_date']).dt.days

    # Handle negative recency (if any dates crept past snapshot)
    df_rfm['recency'] = df_rfm['recency'].apply(lambda x: max(x, 0))

    # b) Apply NTILE(5) scoring using rank(method='first') to handle ties in e-commerce frequency
    df_rfm['R_score'] = pd.qcut(df_rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df_rfm['F_score'] = pd.qcut(df_rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df_rfm['M_score'] = pd.qcut(df_rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # c) Combined scores
    df_rfm['RFM_score'] = (df_rfm['R_score'] * 100) + (df_rfm['F_score'] * 10) + df_rfm['M_score']
    df_rfm['RFM_combined'] = (df_rfm['R_score'] + df_rfm['F_score'] + df_rfm['M_score']) / 3.0

    # d) Segmentation Logic
    def segment_customer(row):
        r, f = row['R_score'], row['F_score']
        if r >= 4 and f >= 4:
            return 'Champions'
        elif f >= 3 and r >= 3:
            return 'Loyal'
        elif r >= 3 and f == 2:
            return 'Potential Loyal'
        elif r >= 4 and f <= 1:
            return 'Recent'
        elif r == 3 and f == 1:
            return 'Promising'
        elif r == 2 and f >= 2:
            return 'Need Attention'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r == 1 and f >= 4:
            return 'Cannot Lose'
        elif r <= 2 and f <= 2 and not (r == 1 and f == 1):
            return 'Hibernating'
        elif r == 1 and f == 1:
            return 'Lost'
        else:
            return 'Other'  # Fallback for logical completeness

    df_rfm['segment'] = df_rfm.apply(segment_customer, axis=1)

    # f) Save and Print
    df_rfm.to_csv('rfm_segments.csv', index=False)

    print("RFM Segment Distribution:")
    dist = df_rfm['segment'].value_counts()
    dist_pct = (dist / len(df_rfm)) * 100
    for seg in dist.index:
        print(f" - {seg}: {dist[seg]:,} ({dist_pct[seg]:.1f}%)")

    return df_rfm


def calculate_clv(df_rfm, conn):
    """
    TASK 2: Calculates Historical and Predictive Customer Lifetime Value.
    """
    print("\n" + "=" * 50)
    print("TASK 2: CUSTOMER LIFETIME VALUE (CLV) CALCULATION")
    print("=" * 50)

    # Fetch first order dates to calculate tenure
    query = """
        SELECT customer_unique_id, MIN(order_purchase_timestamp) as first_order_date
        FROM fact_orders
        GROUP BY customer_unique_id
    """
    first_orders = pd.read_sql_query(query, conn)
    first_orders['first_order_date'] = pd.to_datetime(first_orders['first_order_date'])

    df_rfm = df_rfm.merge(first_orders, on='customer_unique_id', how='left')

    # a) Historical CLV
    df_rfm['historical_clv'] = df_rfm['monetary']

    # Calculate tenure in years (minimum 0.08 years ~ 1 month)
    df_rfm['tenure_days'] = (df_rfm['last_order_date'] - df_rfm['first_order_date']).dt.days
    df_rfm['customer_tenure_years'] = df_rfm['tenure_days'].apply(lambda x: max(x / 365.0, 0.08))

    # b) Predictive CLV components
    df_rfm['avg_order_value'] = df_rfm['monetary'] / df_rfm['frequency']
    df_rfm['purchase_frequency_per_year'] = df_rfm['frequency'] / df_rfm['customer_tenure_years']
    avg_customer_lifespan = 2.0

    df_rfm['predicted_clv'] = df_rfm['avg_order_value'] * df_rfm['purchase_frequency_per_year'] * avg_customer_lifespan

    # Cap at 99th percentile
    cap_value = df_rfm['predicted_clv'].quantile(0.99)
    df_rfm['predicted_clv'] = df_rfm['predicted_clv'].clip(upper=cap_value)

    # c) CLV Tiers
    df_rfm['clv_tier'] = pd.qcut(df_rfm['predicted_clv'].rank(method='first'), 4,
                                 labels=['Tier 4 (Bronze)', 'Tier 3 (Silver)', 'Tier 2 (Gold)', 'Tier 1 (Platinum)'])

    # e) Revenue from Platinum
    total_rev = df_rfm['historical_clv'].sum()
    plat_rev = df_rfm[df_rfm['clv_tier'] == 'Tier 1 (Platinum)']['historical_clv'].sum()
    plat_pct = (plat_rev / total_rev) * 100

    print(f"Top 25% Platinum customers drive {plat_pct:.1f}% of total historical revenue.")

    return df_rfm


def plot_rfm_distribution(df_rfm):
    """TASK 3A: Horizontal bar chart for RFM Segment Distribution"""
    agg_df = df_rfm.groupby('segment').agg(
        count=('customer_unique_id', 'count'),
        avg_score=('RFM_combined', 'mean')
    ).sort_values('count', ascending=True)

    agg_df['pct'] = (agg_df['count'] / agg_df['count'].sum()) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use RdYlGn colormap based on average combined score
    norm = plt.Normalize(agg_df['avg_score'].min(), agg_df['avg_score'].max())
    colors = plt.cm.RdYlGn(norm(agg_df['avg_score']))

    bars = ax.barh(agg_df.index, agg_df['count'], color=colors, edgecolor='black', alpha=0.8)

    for bar, pct in zip(bars, agg_df['pct']):
        width = bar.get_width()
        # Add Count at the end
        ax.text(width + (agg_df['count'].max() * 0.01), bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}", va='center', fontsize=10)
        # Add Percentage inside the bar
        if width > (agg_df['count'].max() * 0.1):  # Only if bar is wide enough
            ax.text(width / 2, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va='center', ha='center', color='black', fontweight='bold')

    plt.title("RFM Segment Distribution (Color = Avg RFM Score)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("rfm_segment_distribution.png", dpi=300)
    plt.close()


def plot_clv_by_segment(df_rfm):
    """TASK 3B: Box Plot of CLV by Segment with Jitter"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Order segments by median CLV for better readability
    order = df_rfm.groupby('segment')['historical_clv'].median().sort_values(ascending=False).index

    sns.boxplot(x='segment', y='historical_clv', data=df_rfm, order=order,
                ax=ax, showfliers=False, palette='Set2')

    # Add jittered dots
    sns.stripplot(x='segment', y='historical_clv', data=df_rfm, order=order,
                  ax=ax, color='black', alpha=0.1, size=2, jitter=True)

    ax.set_yscale('log')
    overall_median = df_rfm['historical_clv'].median()
    ax.axhline(overall_median, color='red', linestyle='--', label=f'Overall Median (₹{overall_median:,.0f})')

    plt.title("Customer Lifetime Value Distribution by RFM Segment", fontsize=15, fontweight='bold')
    plt.ylabel("Historical CLV (₹) - Log Scale")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("clv_by_segment.png", dpi=300)
    plt.close()


def plot_rfm_scatter(df_rfm):
    """TASK 3C: RFM 3D Scatter simulated as 2D"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # To avoid plotting 90k points, we'll sample if dataset is huge, or group
    sample_df = df_rfm.sample(min(10000, len(df_rfm)), random_state=42)

    scatter = sns.scatterplot(
        x='recency', y='monetary', size='frequency', sizes=(20, 400),
        hue='segment', data=sample_df, alpha=0.6, palette='tab10', ax=ax
    )

    # Annotate Champions cluster
    champ_x = sample_df[sample_df['segment'] == 'Champions']['recency'].median()
    champ_y = sample_df[sample_df['segment'] == 'Champions']['monetary'].median()

    if pd.notna(champ_x) and pd.notna(champ_y):
        ax.annotate('Champions\n(Recent, High Spend, High Freq)',
                    xy=(champ_x, champ_y), xytext=(champ_x + 100, champ_y * 2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.title("RFM Customer Universe — Size = Purchase Frequency", fontsize=15, fontweight='bold')
    plt.xlabel("Recency (Days since last purchase)")
    plt.ylabel("Monetary Value (₹)")
    plt.yscale('log')  # Log scale for monetary due to outliers

    # Fix legend
    h, l = scatter.get_legend_handles_labels()
    ax.legend(h[1:11], l[1:11], title="Segments", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("rfm_scatter.png", dpi=300)
    plt.close()


def plot_revenue_concentration(df_rfm):
    """TASK 3D: Lorenz Curve for Revenue Concentration"""
    # Sort and calculate cumsums
    sorted_df = df_rfm.sort_values('monetary').copy()
    sorted_df['cum_customers'] = np.arange(1, len(sorted_df) + 1) / len(sorted_df) * 100
    sorted_df['cum_revenue'] = sorted_df['monetary'].cumsum() / sorted_df['monetary'].sum() * 100

    # Calculate Gini - safely handle numpy 1.x vs 2.x
    y_vals = sorted_df['cum_revenue'] / 100
    x_vals = sorted_df['cum_customers'] / 100
    try:
        auc = np.trapezoid(y_vals, x_vals)
    except AttributeError:
        auc = np.trapz(y_vals, x_vals)

    gini = 1 - 2 * auc

    # Find Top 20% contribution
    # The bottom 80% of customers are at cum_customers = 80, so top 20% revenue = 100 - cum_revenue at 80
    bottom_80_rev = sorted_df[sorted_df['cum_customers'] <= 80]['cum_revenue'].max()
    top_20_contribution = 100 - bottom_80_rev

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(sorted_df['cum_customers'], sorted_df['cum_revenue'], color='#185FA5', linewidth=3, label='Lorenz Curve')
    ax.plot([0, 100], [0, 100], color='gray', linestyle='--', label='Equality Line')
    ax.fill_between(sorted_df['cum_customers'], sorted_df['cum_revenue'], sorted_df['cum_customers'], color='#185FA5',
                    alpha=0.2)

    ax.annotate(f"Top 20% of customers\n= {top_20_contribution:.1f}% of revenue",
                xy=(80, bottom_80_rev), xytext=(50, 80),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))

    ax.text(5, 90, f"Gini Coefficient: {gini:.3f}\n(0=Perfect Equality, 1=Extreme Inequality)",
            fontsize=11, bbox=dict(facecolor='white', edgecolor='gray'))

    plt.title("Revenue Concentration (Lorenz Curve)", fontsize=15, fontweight='bold')
    plt.xlabel("Cumulative % of Customers")
    plt.ylabel("Cumulative % of Revenue")
    plt.legend(loc='lower right')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("revenue_concentration.png", dpi=300)
    plt.close()


def generate_segment_strategy(df_rfm, conn):
    """TASK 4: Creates a comprehensive segment-level business strategy table"""
    print("\n" + "=" * 50)
    print("TASK 4: SEGMENT-LEVEL BUSINESS STRATEGY")
    print("=" * 50)

    # Get avg review scores
    query = "SELECT customer_unique_id, AVG(review_score) as avg_review_score FROM fact_orders GROUP BY customer_unique_id"
    reviews = pd.read_sql_query(query, conn)
    df_merged = df_rfm.merge(reviews, on='customer_unique_id', how='left')

    total_rev = df_merged['monetary'].sum()

    strategy = df_merged.groupby('segment').agg(
        count=('customer_unique_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        avg_clv=('predicted_clv', 'mean'),
        total_revenue=('monetary', 'sum'),
        avg_review_score=('avg_review_score', 'mean')
    ).reset_index()

    strategy['revenue_contribution_pct'] = (strategy['total_revenue'] / total_rev) * 100
    strategy.drop(columns=['total_revenue'], inplace=True)

    # Action mapping
    actions = {
        'Champions': 'Launch VIP loyalty program, request reviews, upsell premium',
        'Loyal': 'Offer loyalty rewards, cross-sell related categories',
        'Potential Loyal': 'Recommend popular combinations, offer membership trial',
        'Recent': 'Send onboarding guide, trigger 2nd purchase discount',
        'Promising': 'Build brand awareness, send personalized recommendations',
        'Need Attention': 'Provide limited-time offers, ask for product feedback',
        'At Risk': 'Send win-back campaign with 15% discount, survey for dissatisfaction',
        'Cannot Lose': 'URGENT: Personal outreach, heavy discount, understand churn reason',
        'Hibernating': 'Send standard reactivation campaign based on past category',
        'Lost': 'Low-cost re-engagement email only, not worth heavy spend',
        'Other': 'General marketing flow'
    }

    strategy['recommended_action'] = strategy['segment'].map(actions)

    # Format for output
    strategy = strategy.round(2)
    strategy.to_csv("segment_strategy.csv", index=False)

    try:
        from tabulate import tabulate
        print(tabulate(
            strategy[['segment', 'count', 'revenue_contribution_pct', 'avg_review_score', 'recommended_action']],
            headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        pd.set_option('display.max_colwidth', 50)
        print(strategy[['segment', 'count', 'revenue_contribution_pct', 'recommended_action']].to_string(index=False))


def rfm_cohort_bridge(df_rfm, conn):
    """TASK 5: Joins RFM with Cohorts and plots the distribution."""
    print("\n" + "=" * 50)
    print("TASK 5: COHORT-RFM BRIDGE")
    print("=" * 50)

    query = """
        SELECT customer_unique_id, MIN(order_year_month) as cohort_month
        FROM fact_orders
        GROUP BY customer_unique_id
    """
    cohorts = pd.read_sql_query(query, conn)

    merged = df_rfm.merge(cohorts, on='customer_unique_id', how='inner')

    # Pivot: % of each cohort belonging to each segment
    pivot = pd.crosstab(merged['cohort_month'], merged['segment'], normalize='index') * 100

    # Keep only the major cohorts to avoid a massive chart
    if len(pivot) > 15:
        pivot = pivot.tail(15)

        # Sort columns to put good segments at the bottom of the stack
    ordered_cols = [c for c in ['Champions', 'Loyal', 'Potential Loyal', 'Recent', 'Promising',
                                'Need Attention', 'At Risk', 'Cannot Lose', 'Hibernating', 'Lost', 'Other'] if
                    c in pivot.columns]
    pivot = pivot[ordered_cols]

    # Identify best month for Champions
    best_champ_month = pivot['Champions'].idxmax() if 'Champions' in pivot.columns else "N/A"
    print(f"Acquisition month producing the highest percentage of Champions: {best_champ_month}")

    # Plot Stacked Bar
    ax = pivot.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')

    plt.title("RFM Segment Distribution by Acquisition Cohort (Last 15 Months)", fontsize=15, fontweight='bold')
    plt.ylabel("% of Customers in Cohort")
    plt.xlabel("Acquisition Cohort")
    plt.legend(title='RFM Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("rfm_cohort_bridge.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    t_start = time.time()
    db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')

    try:
        conn = sqlite3.connect(db_path)

        # Tasks Execution
        t0 = time.time()
        df_rfm = calculate_rfm_scores(conn)
        print(f"[Task 1 Time: {time.time() - t0:.2f}s]")

        t0 = time.time()
        df_rfm = calculate_clv(df_rfm, conn)
        print(f"[Task 2 Time: {time.time() - t0:.2f}s]")

        t0 = time.time()
        plot_rfm_distribution(df_rfm)
        plot_clv_by_segment(df_rfm)
        plot_rfm_scatter(df_rfm)
        plot_revenue_concentration(df_rfm)
        print(f"[Task 3 Visualisations Time: {time.time() - t0:.2f}s]")

        t0 = time.time()
        generate_segment_strategy(df_rfm, conn)
        print(f"[Task 4 Time: {time.time() - t0:.2f}s]")

        t0 = time.time()
        rfm_cohort_bridge(df_rfm, conn)
        print(f"[Task 5 Time: {time.time() - t0:.2f}s]")

        # Summary
        champs = len(df_rfm[df_rfm['segment'] == 'Champions'])
        risk = len(df_rfm[df_rfm['segment'] == 'At Risk'])
        lost = len(df_rfm[df_rfm['segment'] == 'Lost'])

        conn.close()
        elapsed = time.time() - t_start
        print(f"\n✅ Phase 4 Complete! Total Execution Time: {elapsed:.2f} seconds")
        print(f"Segmentation complete. {champs:,} Champions, {risk:,} At Risk, {lost:,} Lost customers identified.")

    except sqlite3.OperationalError as e:
        print(f"\n❌ DB Error: {e}")
        print("Please ensure 'ecommerce_analytics.db' exists in the current directory.")