import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import os
import time
import warnings

# Suppress minor warnings
warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")


def load_funnel_data(conn):
    """Helper function to load and prep order data for funnel analysis."""
    query = """
        SELECT 
            order_id, customer_unique_id, order_status,
            order_purchase_timestamp, order_approved_at, 
            order_delivered_carrier_date, order_delivered_customer_date, 
            review_score, payment_value, product_category_name_english as category,
            customer_region, price_bucket, order_year_month
        FROM fact_orders
    """
    df = pd.read_sql_query(query, conn)

    # Parse datetimes
    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def calculate_master_funnel(conn):
    """
    TASK 1: Calculates absolute counts, conversion rates, and revenue impact for all 6 funnel stages.
    """
    print("\n" + "=" * 50)
    print("TASK 1: MASTER FUNNEL CALCULATION")
    print("=" * 50)

    df = load_funnel_data(conn)

    # Define boolean masks for each sequential stage
    m1 = df['order_id'].notna()
    m2 = m1 & df['order_approved_at'].notna()
    m3 = m2 & df['order_delivered_carrier_date'].notna()
    m4 = m3 & df['order_delivered_customer_date'].notna() & (df['order_status'] == 'delivered')
    m5 = m4 & df['review_score'].notna()

    # For Stage 6, we find customers who reached stage 4 and have > 1 total order
    delivered_customers = df[m4]['customer_unique_id']
    repeat_customers = df['customer_unique_id'].value_counts()
    repeat_customers = repeat_customers[repeat_customers >= 2].index
    m6 = df['customer_unique_id'].isin(repeat_customers) & m4

    masks = [m1, m2, m3, m4, m5, m6]
    stages = [
        '1. Orders Placed', '2. Orders Approved', '3. Orders Dispatched',
        '4. Orders Delivered', '5. Orders Reviewed', '6. Repeat Purchase'
    ]

    funnel_data = []
    prev_count = None
    initial_count = m1.sum()

    for i, mask in enumerate(masks):
        current_df = df[mask]
        count = mask.sum()
        revenue = current_df['payment_value'].sum()

        drop_count = prev_count - count if prev_count is not None else 0
        stage_conv = (count / prev_count * 100) if prev_count else 100.0
        overall_conv = (count / initial_count * 100)

        funnel_data.append({
            'Stage': stages[i],
            'Absolute_Count': count,
            'Drop_From_Previous': drop_count,
            'Stage_Conversion_Rate': stage_conv,
            'Overall_Conversion_Rate': overall_conv,
            'Revenue_At_Stage': revenue,
            'Avg_Order_Value': revenue / count if count > 0 else 0
        })
        prev_count = count

    funnel_df = pd.DataFrame(funnel_data)

    # Calculate revenue lost at each stage
    funnel_df['Revenue_Lost_At_Stage'] = funnel_df['Drop_From_Previous'] * funnel_df['Avg_Order_Value'].shift(1).fillna(
        0)

    # Time between stages (Hours)
    t1_2 = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds().mean() / 3600
    t2_3 = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds().mean() / 3600
    t3_4 = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.total_seconds().mean() / 3600

    print(f"Avg Time 1->2 (Approval): {t1_2:.1f} hours")
    print(f"Avg Time 2->3 (Dispatch): {t2_3:.1f} hours")
    print(f"Avg Time 3->4 (Delivery): {t3_4:.1f} hours")

    # Identify biggest drop-off (excluding stage 1 which has 0 drop)
    biggest_drop_idx = funnel_df.loc[1:, 'Drop_From_Previous'].idxmax()
    worst_stage = funnel_df.loc[biggest_drop_idx]
    prev_stage = funnel_df.loc[biggest_drop_idx - 1]

    print(f"\nBiggest drop-off: {prev_stage['Stage']} → {worst_stage['Stage']} "
          f"with {(100 - worst_stage['Stage_Conversion_Rate']):.1f}% conversion loss "
          f"representing ₹{worst_stage['Revenue_Lost_At_Stage']:,.0f} in revenue.")

    return funnel_df


def plot_conversion_funnel(funnel_df):
    """
    TASK 2: Creates a professional trapezoid funnel chart using matplotlib patches.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    y_points = np.arange(len(funnel_df), 0, -1) * 2  # Space out the Y coordinates
    widths = funnel_df['Overall_Conversion_Rate'].values

    # Create colors from dark blue to light blue
    colors = sns.color_palette("Blues_r", len(funnel_df))

    total_rev_lost = 0
    biggest_opportunity = ""
    max_lost = 0

    for i in range(len(funnel_df)):
        width = widths[i]
        next_width = widths[i + 1] if i < len(funnel_df) - 1 else widths[i] * 0.5  # Taper the bottom

        y_top = y_points[i] + 1
        y_bottom = y_points[i] - 1

        # Trapezoid coordinates
        x_coords = [-width / 2, width / 2, next_width / 2, -next_width / 2]
        y_coords = [y_top, y_top, y_bottom, y_bottom]

        poly = Polygon(xy=list(zip(x_coords, y_coords)), facecolor=colors[i], edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(poly)

        # Annotate inside
        stage_name = funnel_df.iloc[i]['Stage'].split('. ')[1]
        count = funnel_df.iloc[i]['Absolute_Count']
        rate = funnel_df.iloc[i]['Stage_Conversion_Rate']

        rate_color = 'red' if rate < 80 and i > 0 else 'black'
        text_str = f"{stage_name}\n{count:,} orders"
        if i > 0:
            text_str += f"\n{rate:.1f}% from prev"

        ax.text(0, y_points[i], text_str, ha='center', va='center',
                color='white' if i < 3 else 'black', fontweight='bold', fontsize=10)

        # Annotate Revenue Lost outside the funnel
        if i > 0:
            rev_lost = funnel_df.iloc[i]['Revenue_Lost_At_Stage']
            total_rev_lost += rev_lost
            if rev_lost > max_lost:
                max_lost = rev_lost
                biggest_opportunity = f"{funnel_df.iloc[i - 1]['Stage'].split('. ')[1]} → {stage_name}"

            ax.text(width / 2 + 5, y_top, f"- ₹{rev_lost:,.0f} lost",
                    ha='left', va='center', color='red', fontweight='bold', fontsize=9)

    ax.set_xlim(-60, 100)
    ax.set_ylim(0, max(y_points) + 2)
    ax.axis('off')

    plt.title("E-Commerce Order Funnel — Olist Dataset", fontsize=16, fontweight='bold', pad=20)

    # Text box in bottom right
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05,
            f"Total revenue lost in funnel: ₹{total_rev_lost:,.0f}\nBiggest opportunity: {biggest_opportunity}",
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig("conversion_funnel.png", dpi=300)
    plt.close()


def analyse_funnel_by_segment(conn):
    """
    TASK 3: Runs segmented funnel analysis by Region, Price, and Time.
    """
    print("\n" + "=" * 50)
    print("TASK 3: SEGMENTED FUNNEL ANALYSIS")
    print("=" * 50)

    df = load_funnel_data(conn)

    # Breakdown A: Region
    region_data = []
    for region in df['customer_region'].unique():
        if region == 'Unknown': continue
        r_df = df[df['customer_region'] == region]
        s1 = len(r_df)
        s3 = r_df['order_delivered_carrier_date'].notna().sum()
        s4 = (r_df['order_delivered_customer_date'].notna() & (r_df['order_status'] == 'delivered')).sum()
        s5 = r_df[r_df['order_status'] == 'delivered']['review_score'].notna().sum()

        region_data.append({
            'Region': region,
            'Stage 1->4 (Delivery Rate)': (s4 / s1 * 100) if s1 else 0,
            'Stage 3->4 (Carrier to Customer)': (s4 / s3 * 100) if s3 else 0,
            'Stage 4->5 (Review Rate)': (s5 / s4 * 100) if s4 else 0
        })

    reg_df = pd.DataFrame(region_data).set_index('Region')
    worst_delivery_region = reg_df['Stage 3->4 (Carrier to Customer)'].idxmin()
    print(f"Region with worst delivery success (Stage 3->4): {worst_delivery_region}")

    reg_df[['Stage 1->4 (Delivery Rate)', 'Stage 4->5 (Review Rate)']].plot(kind='bar', figsize=(10, 5),
                                                                            color=['#185FA5', '#E67E22'])
    plt.title("Funnel Conversion Rates by Region", fontweight='bold')
    plt.ylabel("Conversion Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("funnel_by_region.png", dpi=300)
    plt.close()

    # Breakdown B: Price Bucket
    price_data = []
    for pb in ['Low', 'Mid', 'High']:
        p_df = df[df['price_bucket'] == pb]
        s4 = (p_df['order_status'] == 'delivered').sum()
        s5 = p_df[p_df['order_status'] == 'delivered']['review_score'].notna().sum()

        cust_counts = p_df['customer_unique_id'].value_counts()
        s6 = (cust_counts >= 2).sum()
        total_cust = len(cust_counts)

        price_data.append({
            'Price Bucket': pb,
            'Review Rate (4->5)': (s5 / s4 * 100) if s4 else 0,
            'Repeat Rate (Stage 6)': (s6 / total_cust * 100) if total_cust else 0
        })

    pd.DataFrame(price_data).set_index('Price Bucket').plot(kind='bar', figsize=(9, 5), color=['#1D9E75', '#8E44AD'])
    plt.title("Post-Purchase Engagement by Price Bucket", fontweight='bold')
    plt.ylabel("Conversion Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("funnel_by_price.png", dpi=300)
    plt.close()

    # Breakdown C: Trend over time
    monthly = []
    for ym, m_df in df.groupby('order_year_month'):
        s1 = len(m_df)
        s2 = m_df['order_approved_at'].notna().sum()
        s4 = (m_df['order_status'] == 'delivered').sum()
        s5 = m_df[m_df['order_status'] == 'delivered']['review_score'].notna().sum()

        if s1 > 50:  # Filter noisy low-volume months
            monthly.append({
                'Month': ym,
                '1->2 (Approval)': (s2 / s1) * 100,
                '2->4 (Delivery)': (s4 / s2) * 100 if s2 else 0,
                '4->5 (Review)': (s5 / s4) * 100 if s4 else 0
            })

    m_trend = pd.DataFrame(monthly).set_index('Month')

    fig, ax = plt.subplots(figsize=(12, 6))
    m_trend.plot(ax=ax, linewidth=2, marker='o')

    # Shaded regions for drops below 1 std dev
    mean_approval = m_trend['1->2 (Approval)'].mean()
    std_approval = m_trend['1->2 (Approval)'].std()
    threshold = mean_approval - std_approval

    for idx, row in m_trend.iterrows():
        if row['1->2 (Approval)'] < threshold:
            ax.axvspan(m_trend.index.get_loc(idx) - 0.5, m_trend.index.get_loc(idx) + 0.5, color='red', alpha=0.2)

    plt.title("Funnel Conversion Trend Over Time", fontweight='bold')
    plt.ylabel("Conversion Rate (%)")
    plt.xticks(range(len(m_trend)), m_trend.index, rotation=45)
    plt.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig("funnel_trend.png", dpi=300)
    plt.close()


def quantify_funnel_opportunity(conn, funnel_df):
    """
    TASK 4: Calculates revenue recovery opportunity and generates a priority matrix.
    """
    print("\n" + "=" * 50)
    print("TASK 4: BUSINESS IMPACT QUANTIFICATION")
    print("=" * 50)

    opportunities = []
    diff_mapping = {1: 1, 2: 2, 3: 3, 4: 1, 5: 3}  # Keys are index in funnel_df

    for i in range(1, len(funnel_df)):
        prev_stage = funnel_df.iloc[i - 1]
        curr_stage = funnel_df.iloc[i]

        current_conv = curr_stage['Stage_Conversion_Rate']
        if current_conv >= 95: continue  # Already highly optimized

        target_conv = current_conv + 5.0
        orders_recovered = prev_stage['Absolute_Count'] * 0.05
        revenue_recovered = orders_recovered * prev_stage['Avg_Order_Value']

        difficulty = diff_mapping.get(i, 2)

        opportunities.append({
            'stage_transition': f"{prev_stage['Stage'].split('. ')[1]} → {curr_stage['Stage'].split('. ')[1]}",
            'current_conversion_pct': current_conv,
            'target_conversion_pct': target_conv,
            'orders_recovered': orders_recovered,
            'revenue_recovered': revenue_recovered,
            'implementation_difficulty': difficulty,
            'priority_score': revenue_recovered / difficulty
        })

    opp_df = pd.DataFrame(opportunities)
    if opp_df.empty:
        print("Funnel is highly optimized. No immediate 5% recovery opportunities found.")
        return

    opp_df.to_csv("funnel_opportunity_matrix.csv", index=False)

    # Bubble chart
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        opp_df['implementation_difficulty'],
        opp_df['revenue_recovered'],
        s=opp_df['orders_recovered'] * 2,  # Scale bubble
        c=opp_df['priority_score'],
        cmap='RdYlGn', alpha=0.7, edgecolors='black'
    )

    for idx, row in opp_df.iterrows():
        ax.annotate(row['stage_transition'],
                    (row['implementation_difficulty'], row['revenue_recovered']),
                    xytext=(0, 15), textcoords='offset points', ha='center', fontweight='bold')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Easy (1)', 'Medium (2)', 'Hard (3)'])
    plt.title("Revenue Recovery Opportunity Matrix (+5% Conversion Target)", fontsize=14, fontweight='bold')
    plt.xlabel("Implementation Difficulty")
    plt.ylabel("Potential Revenue Recovered (₹)")
    plt.colorbar(scatter, label='Priority Score (Rev / Difficulty)')

    plt.tight_layout()
    plt.savefig("funnel_opportunity_matrix.png", dpi=300)
    plt.close()

    # Top recommendation
    top_opp = opp_df.loc[opp_df['priority_score'].idxmax()]

    actions = {
        'Orders Placed → Orders Approved': 'Fix payment gateway timeout errors and review fraud filters.',
        'Orders Approved → Orders Dispatched': 'Optimize warehouse picking SLAs and inventory sync.',
        'Orders Dispatched → Orders Delivered': 'Change 3PL carriers for high-delay zip codes.',
        'Orders Delivered → Orders Reviewed': 'Send automated SMS/Email reminders 2 days post-delivery.',
        'Orders Reviewed → Repeat Purchase': 'Trigger personalized discount cohorts based on first category.'
    }

    action = actions.get(top_opp['stage_transition'], "Conduct deep-dive analysis on this specific drop-off.")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" TOP FUNNEL OPPORTUNITY:")
    print(f" Stage: {top_opp['stage_transition']}")
    print(
        f" Current conversion: {top_opp['current_conversion_pct']:.1f}%  |  Target: {top_opp['target_conversion_pct']:.1f}%")
    print(f" Revenue recoverable: ₹{top_opp['revenue_recovered']:,.0f}")
    print(f" Recommended action: {action}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def analyse_repeat_purchase_behaviour(conn):
    """
    TASK 5: Analyzes timing and product associations for repeat purchases.
    """
    print("\n" + "=" * 50)
    print("TASK 5: REPEAT PURCHASE ANALYSIS")
    print("=" * 50)

    df = load_funnel_data(conn)
    df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])

    # Identify 1st and 2nd orders
    df['order_rank'] = df.groupby('customer_unique_id').cumcount() + 1
    repeats = df[df['order_rank'] <= 2]

    # Pivot to get dates side by side
    pivot_dates = repeats.pivot(index='customer_unique_id', columns='order_rank', values='order_purchase_timestamp')
    pivot_dates = pivot_dates.dropna(subset=[2])  # Keep only those who actually have a 2nd order

    pivot_dates['days_to_second'] = (pivot_dates[2] - pivot_dates[1]).dt.days
    days = pivot_dates['days_to_second']

    # Plot Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(0, 365, 15)
    sns.histplot(days, bins=bins, color='#8E44AD', ax=ax, edgecolor='white')

    median_days = days.median()
    ax.axvline(median_days, color='black', linestyle='--', label=f'Median: {median_days:.0f} days')
    ax.axvline(90, color='red', linestyle='--', label='90 Day Mark')

    ax.axvspan(0, 30, color='green', alpha=0.1, label='High Intent Window (0-30 days)')

    plt.title("Time Between First and Second Purchase", fontsize=14, fontweight='bold')
    plt.xlabel("Days Between Purchases")
    plt.ylabel("Number of Customers")
    plt.xlim(0, 365)
    plt.legend()
    plt.tight_layout()
    plt.savefig("repeat_purchase_timing.png", dpi=300)
    plt.close()

    # Calculate percentages
    within_30 = (days <= 30).sum() / len(days) * 100
    within_60 = (days <= 60).sum() / len(days) * 100
    within_90 = (days <= 90).sum() / len(days) * 100

    # Category Pairs
    cat_pivot = repeats.pivot(index='customer_unique_id', columns='order_rank', values='category').dropna()
    cat_pivot['pair'] = cat_pivot[1] + " + " + cat_pivot[2]
    top_pairs = cat_pivot['pair'].value_counts().head(5)

    print(f"{within_30:.1f}% of customers who ever repurchase do so within 30 days.")
    print(f"{within_60:.1f}% within 60 days, and {within_90:.1f}% within 90 days.")
    print("The optimal re-engagement window is 0-30 days.")

    print("\nTop 5 Category Pairs (First Order + Second Order):")
    for pair, count in top_pairs.items():
        print(f" - {pair}: {count} times")


if __name__ == "__main__":
    start_time = time.time()
    db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')

    try:
        conn = sqlite3.connect(db_path)

        funnel_df = calculate_master_funnel(conn)
        plot_conversion_funnel(funnel_df)
        analyse_funnel_by_segment(conn)
        quantify_funnel_opportunity(conn, funnel_df)
        analyse_repeat_purchase_behaviour(conn)

        conn.close()
        elapsed = time.time() - start_time
        print(f"\n✅ Phase 5 Complete! Total Execution Time: {elapsed:.2f} seconds")

    except sqlite3.OperationalError as e:
        print(f"\n❌ DB Error: {e}")
        print("Please ensure 'ecommerce_analytics.db' exists in the current directory.")