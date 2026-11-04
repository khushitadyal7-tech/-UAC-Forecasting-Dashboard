
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="UAC Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("UAC Care Load & Placement Demand Forecasting")

st.markdown(
    """
    An interactive predictive analytics dashboard for analyzing
    historical trends and forecasting HHS care load, incoming
    placement demand and transfer demand.
    """
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    data = pd.read_csv("weekly_forecasting_data.csv")

    if "Date" not in data.columns:
        data = data.rename(
            columns={data.columns[0]: "Date"}
        )

    data["Date"] = pd.to_datetime(
        data["Date"], errors="coerce"
    )

    data = data.dropna(subset=["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ==========================================
# 3. COLUMN DEFINITIONS
# ==========================================

care_column = "Children in HHS Care"

placement_column = (
    "Children apprehended and placed in CBP custody*"
)

transfer_column = (
    "Children transferred out of CBP custody"
)

required_columns = [
    care_column,
    placement_column,
    transfer_column
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()


for col in required_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )


# ==========================================
# 4. IDENTIFY COMPLETE WEEKS
# ==========================================

if "Coverage" in df.columns:

    complete_weeks = (
        pd.to_numeric(df["Coverage"], errors="coerce") >= 0.999
    )

elif (
    "Expected Days" in df.columns
    and "Observed Expected Days" in df.columns
):

    complete_weeks = (
        (df["Expected Days"] == 5)
        & (df["Observed Expected Days"] == 5)
    )

else:

    complete_weeks = pd.Series(
        True, index=df.index
    )


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),

        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),

        "R2 Score": r2_score(actual, predicted)
    }


def show_trend(data, column, title, ylabel):

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        data["Date"],
        data[column],
        marker=".",
        linewidth=1.7
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()

    st.pyplot(fig)
    plt.close(fig)


def show_forecast(data, title, ylabel):

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        data["Week"],
        data.iloc[:, 1]
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Forecast Week")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=25)

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 6. PREPARE HHS CARE DATA
# ==========================================

def prepare_care_data():

    data = df[["Date", care_column]].copy()

    data["Lag 1"] = data[care_column].shift(1)
    data["Lag 2"] = data[care_column].shift(2)
    data["Lag 4"] = data[care_column].shift(4)

    data["Rolling Mean 4"] = (
        data[care_column].shift(1).rolling(4).mean()
    )

    data["Rolling Std 4"] = (
        data[care_column].shift(1).rolling(4).std()
    )

    data = data.dropna()

    features = [
        "Lag 1",
        "Lag 2",
        "Lag 4",
        "Rolling Mean 4",
        "Rolling Std 4"
    ]

    return data, features


# ==========================================
# 7. PREPARE PLACEMENT AND TRANSFER DATA
# ==========================================

def prepare_flow_data(column):

    data = df[["Date", column]].copy()

    data["Target"] = data[column].where(
        complete_weeks
    )

    data["Lag 1"] = data["Target"].shift(1)
    data["Lag 2"] = data["Target"].shift(2)

    data = data.dropna(
        subset=["Target", "Lag 1", "Lag 2"]
    )

    features = ["Lag 1", "Lag 2"]

    return data, features


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

def evaluate_models(data, features, target):

    X = data[features]
    y = data[target]

    split_point = int(len(data) * 0.8)

    if split_point < 2 or len(data) - split_point < 2:
        raise ValueError(
            "Not enough observations for model evaluation."
        )

    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]

    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]

    results = []

    # Random Forest

    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    results.append({
        "Model": "Random Forest",
        **calculate_metrics(y_test, rf_predictions)
    })

    # Linear Regression

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)

    results.append({
        "Model": "Linear Regression",
        **calculate_metrics(y_test, lr_predictions)
    })

    # Previous Week Baseline

    baseline_predictions = X_test["Lag 1"]

    results.append({
        "Model": "Previous Week Baseline",
        **calculate_metrics(y_test, baseline_predictions)
    })

    return pd.DataFrame(results)


# ==========================================
# 9. FORECAST FUNCTIONS
# ==========================================

def forecast_baseline(last_value, dates):

    return [float(last_value) for _ in dates]


def forecast_random_forest(model, history, dates):

    predictions = []

    values = list(history)

    for date in dates:

        input_data = pd.DataFrame({
            "Lag 1": [values[-1]],
            "Lag 2": [values[-2]]
        })

        prediction = model.predict(input_data)[0]

        prediction = max(0, float(prediction))

        predictions.append(prediction)

        values.append(prediction)

    return predictions


# ==========================================
# 10. PREPARE DATA AND EVALUATE MODELS
# ==========================================

try:

    care_data, care_features = prepare_care_data()

    placement_data, flow_features = prepare_flow_data(
        placement_column
    )

    transfer_data, transfer_features = prepare_flow_data(
        transfer_column
    )

    care_results = evaluate_models(
        care_data,
        care_features,
        care_column
    )

    placement_results = evaluate_models(
        placement_data,
        flow_features,
        "Target"
    )

    transfer_results = evaluate_models(
        transfer_data,
        transfer_features,
        "Target"
    )

except Exception as e:

    st.error(f"Model preparation error: {e}")
    st.stop()


# ==========================================
# 11. GENERATE FOUR-WEEK FORECASTS
# ==========================================

last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(weeks=1),
    periods=4,
    freq="W-SUN"
)

# HHS Care Load: Previous Week Baseline

care_forecast_values = forecast_baseline(
    care_data[care_column].iloc[-1],
    future_dates
)

# Placement Demand: Previous Week Baseline

placement_forecast_values = forecast_baseline(
    placement_data["Target"].iloc[-1],
    future_dates
)

# Transfer Demand: Random Forest trained on all
# available eligible historical observations

transfer_final_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

transfer_final_model.fit(
    transfer_data[transfer_features],
    transfer_data["Target"]
)

transfer_forecast_values = forecast_random_forest(
    transfer_final_model,
    transfer_data["Target"].tolist(),
    future_dates
)


# ==========================================
# 12. CREATE FORECAST TABLES
# ==========================================

care_forecast = pd.DataFrame({
    "Week": future_dates.strftime("%d %b %Y"),
    "Predicted HHS Care Load": np.round(
        care_forecast_values, 2
    )
})

placement_forecast = pd.DataFrame({
    "Week": future_dates.strftime("%d %b %Y"),
    "Predicted Placement Demand": np.round(
        placement_forecast_values, 2
    )
})

transfer_forecast = pd.DataFrame({
    "Week": future_dates.strftime("%d %b %Y"),
    "Predicted Transfer Demand": np.round(
        transfer_forecast_values, 2
    )
})


# ==========================================
# 13. FORECAST DATE EXPLANATION
# ==========================================

forecast_start = future_dates.min().strftime("%d %B %Y")
forecast_end = future_dates.max().strftime("%d %B %Y")

last_observation = last_date.strftime("%d %B %Y")

st.info(
    f"""
    **Forecast information**

    - Last available historical observation: **{last_observation}**
    - Forecast period: **{forecast_start} to {forecast_end}**
    - Forecast horizon: **4 weeks**

    These predictions are calculated using historical weekly data.
    They represent estimates for the four weeks following the
    dataset's last observation. They are not actual recorded values
    and may not represent future dates relative to today.
    """
)


# ==========================================
# 14. SIDEBAR NAVIGATION AND DATE FILTER
# ==========================================

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "HHS Care Load",
        "Placement Demand",
        "Transfer Demand",
        "Model Comparison"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Date Filter")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select historical date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:

    start_date, end_date = date_range

else:

    start_date = min_date
    end_date = max_date


filtered_df = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No records found for the selected date range.")
    st.stop()


# ==========================================
# 15. KPI CARDS
# ==========================================

st.subheader("Project Highlights")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Weekly Records",
        len(filtered_df)
    )

with col2:

    st.metric(
        "Average HHS Care Load",
        f"{filtered_df[care_column].mean():,.0f}"
        if filtered_df[care_column].notna().any()
        else "N/A"
    )

with col3:

    st.metric(
        "Average Placement Demand",
        f"{filtered_df[placement_column].mean():,.0f}"
        if filtered_df[placement_column].notna().any()
        else "N/A"
    )

with col4:

    st.metric(
        "Average Transfer Demand",
        f"{filtered_df[transfer_column].mean():,.0f}"
        if filtered_df[transfer_column].notna().any()
        else "N/A"
    )

st.divider()


# ==========================================
# 16. DASHBOARD OVERVIEW
# ==========================================

if section == "Dashboard Overview":

    st.header("Project Overview")

    st.markdown(
        """
        This project analyzes historical UAC program data
        to understand and forecast:

        - HHS Care Load
        - Incoming Placement Demand
        - Transfer Demand

        Python, Pandas, Matplotlib and Scikit-learn are used
        for data processing, visualization and forecasting.
        """
    )

    st.subheader("Historical Trends")

    show_trend(
        filtered_df,
        care_column,
        "HHS Care Load Trend",
        "Children in HHS Care"
    )

    show_trend(
        filtered_df,
        placement_column,
        "Placement Demand Trend",
        "Number of Placements"
    )

    show_trend(
        filtered_df,
        transfer_column,
        "Transfer Demand Trend",
        "Number of Transfers"
    )

    st.subheader(
        "Four-Week Forecast Based on Last Available Data"
    )

    st.caption(
        f"Forecast period: {forecast_start} to {forecast_end}"
    )

    st.write("HHS Care Load")

    st.dataframe(
        care_forecast,
        use_container_width=True,
        hide_index=True
    )

    st.write("Placement Demand")

    st.dataframe(
        placement_forecast,
        use_container_width=True,
        hide_index=True
    )

    st.write("Transfer Demand")

    st.dataframe(
        transfer_forecast,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 17. HHS CARE LOAD
# ==========================================

elif section == "HHS Care Load":

    st.header("HHS Care Load Analysis")

    show_trend(
        filtered_df,
        care_column,
        "Weekly HHS Care Load",
        "Children in HHS Care"
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        filtered_df[care_column].describe().to_frame(),
        use_container_width=True
    )

    st.subheader("Model Performance")

    st.dataframe(
        care_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Selected forecasting method: Previous Week Baseline"
    )

    st.subheader("Forecast Based on Last Available Data")

    st.dataframe(
        care_forecast,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 18. PLACEMENT DEMAND
# ==========================================

elif section == "Placement Demand":

    st.header("Placement Demand Analysis")

    show_trend(
        filtered_df,
        placement_column,
        "Weekly Placement Demand",
        "Number of Placements"
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        filtered_df[placement_column].describe().to_frame(),
        use_container_width=True
    )

    st.subheader("Model Performance")

    st.dataframe(
        placement_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Selected forecasting method: Previous Week Baseline"
    )

    st.subheader("Forecast Based on Last Available Data")

    st.dataframe(
        placement_forecast,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 19. TRANSFER DEMAND
# ==========================================

elif section == "Transfer Demand":

    st.header("Transfer Demand Analysis")

    show_trend(
        filtered_df,
        transfer_column,
        "Weekly Transfer Demand",
        "Number of Transfers"
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        filtered_df[transfer_column].describe().to_frame(),
        use_container_width=True
    )

    st.subheader("Model Performance")

    st.dataframe(
        transfer_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Selected forecasting method: Random Forest"
    )

    st.subheader("Forecast Based on Last Available Data")

    st.dataframe(
        transfer_forecast,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 20. MODEL COMPARISON
# ==========================================

elif section == "Model Comparison":

    st.header("Model Performance Comparison")

    st.subheader("HHS Care Load")

    st.dataframe(
        care_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Placement Demand")

    st.dataframe(
        placement_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Transfer Demand")

    st.dataframe(
        transfer_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("MAE Comparison")

    comparison = pd.DataFrame({
        "HHS Care Load": care_results.set_index("Model")["MAE"],
        "Placement Demand": placement_results.set_index("Model")["MAE"],
        "Transfer Demand": transfer_results.set_index("Model")["MAE"]
    })

    fig, ax = plt.subplots(figsize=(12, 5))

    comparison.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("MAE Comparison Across Forecasting Tasks")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Final Model Selection")

    st.markdown(
        """
        - **HHS Care Load:** Previous Week Baseline
        - **Placement Demand:** Previous Week Baseline
        - **Transfer Demand:** Random Forest
        """
    )

    st.caption(
        "Metrics are recalculated using an 80/20 chronological "
        "train-test split. Forecast models use the selected "
        "historical observations available for training."
    )


# ==========================================
# 21. DOWNLOAD FORECASTS
# ==========================================

st.sidebar.markdown("---")

st.sidebar.subheader("Download Forecasts")

all_forecasts = pd.DataFrame({
    "Week": future_dates.strftime("%d %b %Y"),
    "Predicted HHS Care Load": care_forecast_values,
    "Predicted Placement Demand": placement_forecast_values,
    "Predicted Transfer Demand": transfer_forecast_values
})

csv_data = all_forecasts.to_csv(
    index=False
).encode("utf-8")

st.sidebar.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name="UAC_Four_Week_Forecasts.csv",
    mime="text/csv"
)


# ==========================================
# 22. FOOTER
# ==========================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "UAC Predictive Forecasting Project"
)

st.sidebar.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit"
)

st.caption(
    "Note: Forecasts are estimates based on historical patterns. "
    "They should not be interpreted as guaranteed outcomes."
)
