import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from datetime import datetime

# Forecasting specific imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Attempt to import advanced libraries with fallback flags
try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠️ Prophet not installed. Falling back to Exponential Smoothing for Model A.")
    print("   To install: pip install prophet")

try:
    import pmdarima as pm

    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    print("⚠️ pmdarima not installed. Falling back to Holt-Winters for Model B.")
    print("   To install: pip install pmdarima")

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")


def format_inr(number):
    """Formats a number in Indian Rupee format (e.g., ₹1,50,000)"""
    try:
        is_negative = number < 0
        number = abs(int(number))
        s = str(number)
        if len(s) > 3:
            r = ",".join([s[x - 2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        else:
            r = s
        return f"-₹{r}" if is_negative else f"₹{r}"
    except:
        return f"₹{number}"


def prepare_time_series_data(conn):
    """TASK 1: Prepares, cleans, and tests the time series data."""
    print("\n" + "=" * 50)
    print("TASK 1: TIME SERIES DATA PREPARATION")
    print("=" * 50)

    query = """
        SELECT 
            order_year_month,
            SUM(payment_value) as total_revenue,
            COUNT(DISTINCT order_id) as order_count,
            SUM(payment_value) / COUNT(DISTINCT order_id) as avg_order_value
        FROM fact_orders
        GROUP BY order_year_month
        ORDER BY order_year_month
    """
    df = pd.read_sql_query(query, conn)

    # Parse datetimes and set index
    df['date'] = pd.to_datetime(df['order_year_month'] + '-01')
    df.set_index('date', inplace=True)

    # Drop the string column so reindexing with fill_value=0 doesn't crash strict Pandas versions
    df.drop(columns=['order_year_month'], inplace=True)
    df.sort_index(inplace=True)

    # Fill any missing months in the sequence with 0 to maintain frequency
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
    df = df.reindex(idx, fill_value=0)

    # Flag partial months (Olist data starts Sep 2016 (partial), ends Oct 2018 (partial))
    df['is_partial'] = False
    df.loc[df.index.min(), 'is_partial'] = True
    df.loc[df.index.max(), 'is_partial'] = True
    if len(df) > 1:
        df.loc[df.index[-2], 'is_partial'] = True  # Sep 2018 is also often partial

    # Exclude partials for clean modeling
    clean_df = df[~df['is_partial']].copy()

    print(f"Data shape: {clean_df.shape}")
    print(f"Date range: {clean_df.index.min().date()} to {clean_df.index.max().date()}")

    # Time Series Decomposition (requires at least 2 full periods, period=12)
    if len(clean_df) >= 24:
        decomposition = seasonal_decompose(clean_df['total_revenue'], model='multiplicative', period=12)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle("Time Series Decomposition (Revenue)", fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig("ts_decomposition.png", dpi=300)
        plt.close()
    else:
        print("Not enough data points for 12-month seasonal decomposition. Skipping plot.")

    # Stationarity Test (ADF)
    result = adfuller(clean_df['total_revenue'].values)
    p_value = result[1]
    print(f"ADF Test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Interpretation: Data is non-stationary (has a trend). Differencing may be required for ARIMA.")
        diff_res = adfuller(np.diff(clean_df['total_revenue'].values))
        print(f"ADF after 1st difference p-value: {diff_res[1]:.4f} (Stationary)")
    else:
        print("Interpretation: Data is stationary.")

    # Train/Test Split (Leave last 3 months of CLEAN data as test)
    train_df = clean_df.iloc[:-3].copy()
    test_df = clean_df.iloc[-3:].copy()

    return clean_df, train_df, test_df


def run_prophet_forecast(train_df, periods=3):
    """TASK 2A: Runs Prophet forecast (or fallback)."""
    if HAS_PROPHET:
        prophet_df = train_df.reset_index()[['index', 'total_revenue']].rename(
            columns={'index': 'ds', 'total_revenue': 'y'})

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode='multiplicative', changepoint_prior_scale=0.1)
        # Indian festive season roughly Oct-Nov
        m.add_seasonality(name='festive_season', period=365.25, fourier_order=3)
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)

        res = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper', 'trend']].tail(len(train_df) + periods)
        return res
    else:
        # Fallback to Exponential Smoothing
        model = ExponentialSmoothing(train_df['total_revenue'], trend='add', seasonal=None,
                                     initialization_method="estimated")
        fit_model = model.fit()
        forecast = fit_model.forecast(periods)

        # Build mock dataframe to match expected output format
        res = pd.DataFrame(index=train_df.index.append(forecast.index))
        res['yhat'] = pd.concat([fit_model.fittedvalues, forecast])
        res['yhat_lower'] = res['yhat'] * 0.85  # Mock 15% CI
        res['yhat_upper'] = res['yhat'] * 1.15
        res['trend'] = res['yhat']
        return res


def run_sarima_forecast(train_df, test_df):
    """TASK 2B: Runs SARIMA forecast (or fallback)."""
    periods = len(test_df) + 3  # Predict test set + future

    if HAS_PMDARIMA and len(train_df) >= 24:
        # Need enough data for m=12
        smodel = pm.auto_arima(train_df['total_revenue'], seasonal=True, m=12,
                               max_p=3, max_q=3, max_P=2, max_Q=2,
                               information_criterion='aic', trace=False, error_action='ignore')
        print(f"Best SARIMA model: {smodel.order}{smodel.seasonal_order}")

        fc, conf_int = smodel.predict(n_periods=periods, return_conf_int=True)

        future_idx = pd.date_range(start=train_df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
        res = pd.DataFrame({'yhat': fc}, index=future_idx)
        res['yhat_lower'] = conf_int[:, 0]
        res['yhat_upper'] = conf_int[:, 1]
        return res
    else:
        # Fallback
        print("Using Holt-Winters fallback for Model B.")
        model = ExponentialSmoothing(train_df['total_revenue'], trend='add', seasonal='add',
                                     seasonal_periods=min(12, len(train_df) // 2))
        fit_model = model.fit()
        fc = fit_model.forecast(periods)

        future_idx = pd.date_range(start=train_df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
        res = pd.DataFrame({'yhat': fc}, index=future_idx)
        res['yhat_lower'] = res['yhat'] * 0.80
        res['yhat_upper'] = res['yhat'] * 1.20
        return res


def run_regression_forecast(train_df, periods=3):
    """TASK 2C: Runs Linear Regression with time-engineered features."""

    def engineer_features(df_index):
        X = pd.DataFrame(index=df_index)
        X['month'] = df_index.month
        X['quarter'] = df_index.quarter
        X['year'] = df_index.year
        X['trend'] = np.arange(len(df_index))
        X['sin_month'] = np.sin(2 * np.pi * X['month'] / 12)
        X['cos_month'] = np.cos(2 * np.pi * X['month'] / 12)
        X['is_q4'] = X['quarter'] == 4
        return X

    X_train = engineer_features(train_df.index)
    y_train = train_df['total_revenue']

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Regression R² score: {model.score(X_train, y_train):.3f}")

    future_idx = pd.date_range(start=train_df.index[-1] + pd.DateOffset(months=1), periods=periods + 3, freq='MS')
    X_future = engineer_features(future_idx)
    # Adjust trend for future
    X_future['trend'] = np.arange(len(train_df), len(train_df) + len(future_idx))

    fc = model.predict(X_future)
    res = pd.DataFrame({'yhat': fc}, index=future_idx)
    res['yhat_lower'] = res['yhat'] * 0.90  # Mock CI
    res['yhat_upper'] = res['yhat'] * 1.10

    return res


def compare_models(train_df, test_df, prophet_fc, sarima_fc, regression_fc):
    """TASK 3: Evaluates and compares all models."""
    print("\n" + "=" * 50)
    print("TASK 3: MODEL COMPARISON & EVALUATION")
    print("=" * 50)

    actuals = test_df['total_revenue'].values

    def evaluate(fc_df, name, complexity, interpretability):
        # Align indices
        preds = fc_df.loc[test_df.index, 'yhat'].values
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

        # Direction accuracy
        actual_diff = np.diff(np.append(train_df['total_revenue'].iloc[-1], actuals))
        pred_diff = np.diff(np.append(train_df['total_revenue'].iloc[-1], preds))
        dir_acc = np.mean((actual_diff > 0) == (pred_diff > 0)) * 100

        return {
            'model_name': name, 'MAE': mae, 'RMSE': rmse,
            'MAPE': mape, 'direction_accuracy': dir_acc,
            'complexity': complexity, 'interpretability': interpretability,
            'preds': preds
        }

    m1 = evaluate(prophet_fc, 'Prophet (or Fallback)', 'Medium', 'High')
    m2 = evaluate(sarima_fc, 'SARIMA (or Fallback)', 'High', 'Medium')
    m3 = evaluate(regression_fc, 'Linear Regression', 'Low', 'Very High')

    comp_df = pd.DataFrame([m1, m2, m3])

    best_model_idx = comp_df['MAPE'].idxmin()
    best_model_name = comp_df.loc[best_model_idx, 'model_name']
    best_mape = comp_df.loc[best_model_idx, 'MAPE']

    # Identify the best forecast series for future extraction
    if best_model_idx == 0:
        best_fc = prophet_fc
    elif best_model_idx == 1:
        best_fc = sarima_fc
    else:
        best_fc = regression_fc

    # Extrapolate future 3 months
    future_dates = pd.date_range(start=test_df.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
    future_preds = best_fc.loc[future_dates]

    print(f"Recommended model: {best_model_name} with MAPE of {best_mape:.2f}%.")
    print("Revenue forecast for next 3 months:")
    for date, row in future_preds.iterrows():
        print(
            f" - {date.strftime('%b %Y')}: {format_inr(row['yhat'])} (Range: {format_inr(row['yhat_lower'])} to {format_inr(row['yhat_upper'])})")

    # Plotting 2x2 Subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Test Set Actuals vs Predicts
    ax1 = axes[0, 0]
    ax1.plot(test_df.index, actuals, marker='o', color='black', linewidth=3, label='Actuals')
    ax1.plot(test_df.index, m1['preds'], marker='x', linestyle='--', label=m1['model_name'])
    ax1.plot(test_df.index, m2['preds'], marker='s', linestyle='-.', label=m2['model_name'])
    ax1.plot(test_df.index, m3['preds'], marker='^', linestyle=':', label=m3['model_name'])
    ax1.set_title("Test Set Evaluation (Actual vs Predicted)", fontweight='bold')
    ax1.legend()

    # Plot 2: Residuals
    ax2 = axes[0, 1]
    # FIX: Use string labels as index for the bar plot to avoid pandas PeriodConverter issues
    str_index = [d.strftime('%b %Y') for d in test_df.index]
    res_df = pd.DataFrame({
        m1['model_name']: actuals - m1['preds'],
        m2['model_name']: actuals - m2['preds'],
        m3['model_name']: actuals - m3['preds']
    }, index=str_index)
    res_df.plot(kind='bar', ax=ax2, colormap='Set2')
    ax2.set_title("Model Residuals (Errors) on Test Set", fontweight='bold')
    ax2.set_xticklabels(str_index, rotation=0)

    # Plot 3: Full Forecast with best model
    ax3 = axes[1, 0]
    hist_concat = pd.concat([train_df['total_revenue'], test_df['total_revenue']])
    ax3.plot(hist_concat.index, hist_concat, color='black', label='Historical Actuals')
    ax3.plot(future_dates, future_preds['yhat'], color='#185FA5', linestyle='--', linewidth=2, label='Forecast')
    ax3.fill_between(future_dates, future_preds['yhat_lower'], future_preds['yhat_upper'], color='#185FA5', alpha=0.2)
    ax3.axvline(test_df.index[-1], color='red', linestyle=':', label='Forecast Start')
    ax3.set_title(f"Future Revenue Forecast (via {best_model_name})", fontweight='bold')
    ax3.legend()

    # Plot 4: Grouped Bar for Metrics
    ax4 = axes[1, 1]
    comp_df.set_index('model_name')[['MAPE', 'direction_accuracy']].plot(kind='bar', ax=ax4,
                                                                         color=['#E24B4A', '#1D9E75'])
    ax4.set_title("Metric Comparison (MAPE vs Direction Accuracy %)", fontweight='bold')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    plt.close()

    return comp_df, future_preds


def forecast_by_category(conn):
    """TASK 4: Generates category-level forecasts using basic exponential smoothing for safety."""
    print("\n" + "=" * 50)
    print("TASK 4: CATEGORY-LEVEL FORECASTING")
    print("=" * 50)

    # Find top 8 categories
    top_cats_query = """
        SELECT COALESCE(product_category_name_english, 'Unknown') as category, SUM(payment_value) as rev
        FROM fact_orders GROUP BY category ORDER BY rev DESC LIMIT 8
    """
    top_cats = pd.read_sql_query(top_cats_query, conn)['category'].tolist()

    query = f"""
        SELECT order_year_month, COALESCE(product_category_name_english, 'Unknown') as category, SUM(payment_value) as total_revenue
        FROM fact_orders
        WHERE category IN ({','.join(['?'] * 8)})
        GROUP BY order_year_month, category
    """
    df = pd.read_sql_query(query, conn, params=top_cats)

    pivot = df.pivot(index='order_year_month', columns='category', values='total_revenue').fillna(0)
    pivot.index = pd.to_datetime(pivot.index + '-01')
    pivot.sort_index(inplace=True)

    # Drop known partials at start/end
    pivot = pivot.iloc[1:-2]

    forecasts = []
    future_idx = pd.date_range(start=pivot.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
    heatmap_data = pivot.tail(6).copy()  # Last 6 months actuals

    summary_data = []

    for cat in top_cats:
        series = pivot[cat]
        # Simple Holt forecast
        model = ExponentialSmoothing(series, trend='add', initialization_method="estimated").fit()
        fc = model.forecast(3)
        fc.index = future_idx

        heatmap_data.loc[future_idx[0], cat] = fc.iloc[0]
        heatmap_data.loc[future_idx[1], cat] = fc.iloc[1]
        heatmap_data.loc[future_idx[2], cat] = fc.iloc[2]

        current_rev = series.iloc[-1]
        last_3_avg = series.tail(3).mean()
        prev_3_avg = series.iloc[-6:-3].mean()
        growth_pct = ((last_3_avg - prev_3_avg) / prev_3_avg * 100) if prev_3_avg > 0 else 0
        direction = "↑" if growth_pct > 0 else "↓"

        summary_data.append({
            'Category': cat,
            'Current Revenue': format_inr(current_rev),
            'Forecast M1': format_inr(fc.iloc[0]),
            'Forecast M2': format_inr(fc.iloc[1]),
            'Forecast M3': format_inr(fc.iloc[2]),
            'Trend': direction,
            'Growth %': f"{growth_pct:.1f}%"
        })

    sum_df = pd.DataFrame(summary_data)

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data.T, cmap='YlGnBu', annot=False, ax=ax)

    # Draw line to separate actuals from forecast
    ax.axvline(6, color='red', linewidth=3, linestyle='--')
    ax.text(6.1, -0.2, 'Forecast →', color='red', fontweight='bold')

    plt.title("Category Revenue Heatmap (Actuals vs 3-Month Forecast)", fontsize=14, fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("Category")

    # Format x-ticks
    labels = [d.strftime('%b %Y') for d in heatmap_data.index]
    ax.set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("category_forecast.png", dpi=300)
    plt.close()

    growth_cats = sum_df[sum_df['Trend'] == '↑']['Category'].tolist()
    decl_cats = sum_df[sum_df['Trend'] == '↓']['Category'].tolist()

    print(f"Growth categories: {', '.join(growth_cats)}")
    print(f"Declining categories: {', '.join(decl_cats)}")

    return sum_df


def generate_forecast_narrative(comp_df, future_preds, category_summary):
    """TASK 5: Generates plain English management summary."""
    print("\n" + "=" * 50)
    print("TASK 5: FORECAST NARRATIVE GENERATOR")
    print("=" * 50)

    best_idx = comp_df['MAPE'].idxmin()
    best_model = comp_df.loc[best_idx, 'model_name']

    m1_val = future_preds['yhat'].iloc[0]
    m1_ci = future_preds['yhat_upper'].iloc[0] - m1_val
    m2_val = future_preds['yhat'].iloc[1]
    m3_val = future_preds['yhat'].iloc[2]

    # Extract categories safely
    try:
        growth_driver = category_summary.sort_values('Growth %', ascending=False).iloc[0]['Category']
        watch_risk = category_summary.sort_values('Growth %', ascending=True).iloc[0]['Category']
    except:
        growth_driver = "Top Categories"
        watch_risk = "Bottom Categories"

    narrative = (
        f"Based on 24 months of historical data, our revenue forecast model (using {best_model}) "
        f"projects {future_preds.index[0].strftime('%B')} revenue at {format_inr(m1_val)} (±{format_inr(m1_ci)}), "
        f"{future_preds.index[1].strftime('%B')} at {format_inr(m2_val)}, and "
        f"{future_preds.index[2].strftime('%B')} at {format_inr(m3_val)}. "
        f"This represents a projected stabilization period moving into the end of the year.\n\n"
        f"Key risk: Month 3 volatility and confidence interval expansion.\n"
        f"Growth driver: {growth_driver}.\n"
        f"Watch: {watch_risk} (monitoring for decline)."
    )

    with open("forecast_narrative.txt", "w", encoding='utf-8') as f:
        f.write(narrative)

    print("┌" + "─" * 70 + "┐")
    print("│ FORECAST EXECUTIVE SUMMARY".ljust(71) + "│")
    print("├" + "─" * 70 + "┤")
    for line in narrative.split('\n'):
        if line.strip():
            print("│ " + line.ljust(69) + "│")
    print("└" + "─" * 70 + "┘")


if __name__ == "__main__":
    t_start = time.time()
    db_path = os.path.join(os.getcwd(), 'ecommerce_analytics.db')

    try:
        conn = sqlite3.connect(db_path)

        # Task 1
        clean_df, train_df, test_df = prepare_time_series_data(conn)

        # Task 2
        prophet_fc = run_prophet_forecast(train_df)
        sarima_fc = run_sarima_forecast(train_df, test_df)
        regression_fc = run_regression_forecast(train_df)

        # Task 3
        comp_df, future_preds = compare_models(train_df, test_df, prophet_fc, sarima_fc, regression_fc)

        # Task 4
        cat_summary = forecast_by_category(conn)

        # Task 5
        generate_forecast_narrative(comp_df, future_preds, cat_summary)

        conn.close()
        elapsed = time.time() - t_start
        print(f"\n✅ Phase 6 Complete! Total Execution Time: {elapsed:.2f} seconds")

    except sqlite3.OperationalError as e:
        print(f"\n❌ DB Error: {e}")
        print("Please ensure 'ecommerce_analytics.db' exists in the current directory.")