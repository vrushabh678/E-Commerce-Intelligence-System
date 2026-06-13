import pandas as pd
import sqlite3
import os
import time
import numpy as np

# Set display options for better terminal output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_and_merge_data(base_path: str) -> pd.DataFrame:
    """
    TASK 1: Loads all 9 Olist CSVs with explicit dtypes and merges them into a master analytical dataframe.
    """
    print("\n" + "=" * 50)
    print("TASK 1: LOADING AND MERGING DATA")
    print("=" * 50)

    try:
        print("Loading CSV files with explicit dtypes...")

        # 1. Orders
        orders = pd.read_csv(
            os.path.join(base_path, 'olist_orders_dataset.csv'),
            dtype={'order_id': 'string', 'customer_id': 'string', 'order_status': 'category'},
            parse_dates=['order_purchase_timestamp', 'order_approved_at',
                         'order_delivered_carrier_date', 'order_delivered_customer_date',
                         'order_estimated_delivery_date']
        )

        # 2. Order Items
        order_items = pd.read_csv(
            os.path.join(base_path, 'olist_order_items_dataset.csv'),
            dtype={'order_id': 'string', 'order_item_id': 'int32', 'product_id': 'string',
                   'seller_id': 'string', 'price': 'float64', 'freight_value': 'float64'},
            parse_dates=['shipping_limit_date']
        )

        # 3. Order Payments (Aggregated per order)
        payments = pd.read_csv(
            os.path.join(base_path, 'olist_order_payments_dataset.csv'),
            dtype={'order_id': 'string', 'payment_sequential': 'int32', 'payment_type': 'category',
                   'payment_installments': 'int32', 'payment_value': 'float64'}
        )
        payments_agg = payments.groupby('order_id', as_index=False)['payment_value'].sum()

        # 4. Order Reviews (Max score per order)
        reviews = pd.read_csv(
            os.path.join(base_path, 'olist_order_reviews_dataset.csv'),
            dtype={'review_id': 'string', 'order_id': 'string', 'review_score': 'float64'},
            parse_dates=['review_creation_date', 'review_answer_timestamp']
        )
        reviews_agg = reviews.groupby('order_id', as_index=False)['review_score'].max()

        # 5. Customers
        customers = pd.read_csv(
            os.path.join(base_path, 'olist_customers_dataset.csv'),
            dtype={'customer_id': 'string', 'customer_unique_id': 'string',
                   'customer_zip_code_prefix': 'string', 'customer_city': 'string', 'customer_state': 'string'}
        )

        # 6. Products
        products = pd.read_csv(
            os.path.join(base_path, 'olist_products_dataset.csv'),
            dtype={'product_id': 'string', 'product_category_name': 'string', 'product_name_lenght': 'float64',
                   'product_description_lenght': 'float64', 'product_photos_qty': 'float64',
                   'product_weight_g': 'float64', 'product_length_cm': 'float64',
                   'product_height_cm': 'float64', 'product_width_cm': 'float64'}
        )

        # 7. Sellers
        sellers = pd.read_csv(
            os.path.join(base_path, 'olist_sellers_dataset.csv'),
            dtype={'seller_id': 'string', 'seller_zip_code_prefix': 'string',
                   'seller_city': 'string', 'seller_state': 'string'}
        )

        # 8. Product Category Translation
        translation = pd.read_csv(
            os.path.join(base_path, 'product_category_name_translation.csv'),
            dtype={'product_category_name': 'string', 'product_category_name_english': 'string'}
        )

        # Merging Pipeline
        print("Executing table merges...")
        df_master = orders.merge(order_items, on='order_id', how='left')
        df_master = df_master.merge(payments_agg, on='order_id', how='left')
        df_master = df_master.merge(reviews_agg, on='order_id', how='left')
        df_master = df_master.merge(customers, on='customer_id', how='left')

        products_translated = products.merge(translation, on='product_category_name', how='left')
        df_master = df_master.merge(products_translated, on='product_id', how='left')
        df_master = df_master.merge(sellers, on='seller_id', how='left')

        print(f"Merge complete. Master DataFrame shape: {df_master.shape}")

        export_path = os.path.join(base_path, 'master_dataset.csv')
        df_master.to_csv(export_path, index=False)
        print(f"Master dataset exported to {export_path}")

        return df_master

    except FileNotFoundError as e:
        print(f"ERROR: {e}. Please ensure all Kaggle CSVs are extracted in the '{base_path}' folder.")
        raise


def generate_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 2: Generates a comprehensive data quality report for the master dataframe.
    """
    print("\n" + "=" * 50)
    print("TASK 2: DATA QUALITY REPORT")
    print("=" * 50)

    dq_data = []
    total_rows = len(df)

    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / total_rows) * 100
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)

        try:
            min_val = df[col].min()
            max_val = df[col].max()
        except TypeError:
            min_val, max_val = "N/A", "N/A"

        sample_vals = df[col].dropna().sample(min(3, unique_count)).tolist() if unique_count > 0 else []

        dq_data.append({
            'Column': col,
            'Data_Type': dtype,
            'Null_Count': null_count,
            'Null_Pct': null_pct,
            'Unique_Count': unique_count,
            'Min_Value': min_val,
            'Max_Value': max_val,
            'Sample_Values': sample_vals
        })

    dq_report = pd.DataFrame(dq_data)

    print("\n--- DATA QUALITY ALERTS ---")
    high_missing = dq_report[dq_report['Null_Pct'] > 5.0]['Column'].tolist()
    if high_missing:
        print(f"[FLAG - HIGH MISSING >5%]: {', '.join(high_missing)}")

    zero_variance = dq_report[dq_report['Unique_Count'] == 1]['Column'].tolist()
    if zero_variance:
        print(f"[FLAG - ZERO VARIANCE]: {', '.join(zero_variance)}")

    full_dupes = df.duplicated().sum()
    print(f"Full duplicate rows: {full_dupes}")

    # Check date range
    if 'order_purchase_timestamp' in df.columns:
        min_date = df['order_purchase_timestamp'].min()
        max_date = df['order_purchase_timestamp'].max()
        print(f"Order Purchase Date Range: {min_date} to {max_date}")

    # Order status distribution
    if 'order_status' in df.columns:
        print("\nOrder Status Distribution (%):")
        print((df['order_status'].value_counts(normalize=True) * 100).round(2).to_string())

    dq_report.to_csv('dq_report.csv', index=False)
    print("\nDetailed Data Quality Report saved to 'dq_report.csv'.")
    return dq_report


def clean_master_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 3: Cleans the dataset, filters for delivered orders, and engineers core analytical features.
    """
    print("\n" + "=" * 50)
    print("TASK 3: DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 50)

    initial_rows = len(df)

    # a) Filter to DELIVERED orders
    print(
        "Filtering to 'delivered' orders. Reason: Cancelled/unavailable orders distort revenue and retention analysis.")
    df_clean = df[df['order_status'] == 'delivered'].copy()

    # b) Remove null purchase timestamps
    df_clean = df_clean.dropna(subset=['order_purchase_timestamp'])

    # c) Feature Engineering
    print("Engineering features: order_year_month, delivery_days, price_bucket, customer_region...")

    df_clean['order_year_month'] = df_clean['order_purchase_timestamp'].dt.strftime('%Y-%m')

    # Calculate delivery days
    delivery_delta = df_clean['order_delivered_customer_date'] - df_clean['order_purchase_timestamp']
    df_clean['delivery_days'] = delivery_delta.dt.days.fillna(-1).astype(int)

    df_clean['is_late_delivery'] = df_clean['delivery_days'] > 10

    # Price bucket mapping
    def map_price(val):
        if pd.isna(val): return 'Unknown'
        if val < 50:
            return 'Low'
        elif val <= 200:
            return 'Mid'
        else:
            return 'High'

    df_clean['price_bucket'] = df_clean['payment_value'].apply(map_price)

    # Geographic mapping
    region_map = {
        'AM': 'North', 'RR': 'North', 'AP': 'North', 'PA': 'North', 'TO': 'North', 'RO': 'North', 'AC': 'North',
        'MA': 'Northeast', 'PI': 'Northeast', 'CE': 'Northeast', 'RN': 'Northeast', 'PB': 'Northeast',
        'PE': 'Northeast', 'AL': 'Northeast', 'SE': 'Northeast', 'BA': 'Northeast',
        'MT': 'Central-West', 'MS': 'Central-West', 'GO': 'Central-West', 'DF': 'Central-West',
        'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
        'PR': 'South', 'SC': 'South', 'RS': 'South'
    }
    df_clean['customer_region'] = df_clean['customer_state'].map(region_map).fillna('Unknown')

    # d) Fill monetary nulls
    df_clean['payment_value'] = df_clean['payment_value'].fillna(0.0)
    df_clean['price'] = df_clean['price'].fillna(0.0)
    df_clean['freight_value'] = df_clean['freight_value'].fillna(0.0)

    final_rows = len(df_clean)
    print(
        f"Cleaned dataset: {final_rows} rows, {len(df_clean.columns)} columns. Removed {initial_rows - final_rows} rows during cleaning.")

    return df_clean


def build_star_schema_db(df: pd.DataFrame, db_name: str = 'ecommerce_analytics.db') -> None:
    """
    TASK 4: Creates a SQLite star schema database from the cleaned master dataframe.
    """
    print("\n" + "=" * 50)
    print("TASK 4: STAR SCHEMA SQL DATABASE CREATION")
    print("=" * 50)

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        print(f"Connected to SQLite database: {db_name}")

        # Fact Table: Orders
        print("Loading fact_orders table...")
        df.to_sql('fact_orders', conn, if_exists='replace', index=False)

        # Dimension: Customers
        print("Building dim_customers...")
        dim_customers = df[
            ['customer_unique_id', 'customer_city', 'customer_state', 'customer_region']].drop_duplicates()
        dim_customers.to_sql('dim_customers', conn, if_exists='replace', index=False)

        # Dimension: Products
        print("Building dim_products...")
        dim_products = df[
            ['product_id', 'product_category_name_english', 'product_weight_g', 'price_bucket']].drop_duplicates()
        dim_products.to_sql('dim_products', conn, if_exists='replace', index=False)

        # Dimension: Sellers
        print("Building dim_sellers...")
        dim_sellers = df[['seller_id', 'seller_city', 'seller_state']].drop_duplicates()
        dim_sellers.to_sql('dim_sellers', conn, if_exists='replace', index=False)

        # Dimension: Time
        print("Building dim_time...")
        df['order_year'] = df['order_purchase_timestamp'].dt.year
        df['order_month'] = df['order_purchase_timestamp'].dt.month
        df['order_quarter'] = df['order_purchase_timestamp'].dt.quarter
        dim_time = df[['order_year_month', 'order_year', 'order_month', 'order_quarter']].drop_duplicates()
        dim_time.to_sql('dim_time', conn, if_exists='replace', index=False)

        # Create Indexes for query performance
        print("Creating database indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON fact_orders(order_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_unique ON fact_orders(customer_unique_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prod_id ON fact_orders(product_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_ym ON fact_orders(order_year_month);")

        # Verification
        print("\n--- DATABASE LOAD VERIFICATION ---")
        tables = ['fact_orders', 'dim_customers', 'dim_products', 'dim_sellers', 'dim_time']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table} row count: {count:,}")

        conn.commit()
        conn.close()
        print("Database connection closed successfully.")

    except Exception as e:
        print(f"DATABASE ERROR: {e}")


if __name__ == "__main__":
    start_time = time.time()

    BASE_PATH = os.path.join(os.getcwd(), 'data')

    # Run the ETL pipeline
    df_raw = load_and_merge_data(BASE_PATH)
    generate_data_quality_report(df_raw)
    df_clean = clean_master_df(df_raw)
    build_star_schema_db(df_clean)

    elapsed = time.time() - start_time
    print(f"\n✅ Phase 1 Complete! Total Execution Time: {elapsed:.2f} seconds")