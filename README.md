# Predictive Forecasting of Care Load & Placement Demand

## Project Overview

This project analyzes historical data from the U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program.

The objective is to understand historical trends and forecast three important measures:

- HHS Care Load
- Incoming Placement Demand
- Transfer Demand

An interactive dashboard was developed using Python and Streamlit to visualize historical data, compare forecasting models, and display four-week predictions.
## Live Demo

Explore the interactive dashboard here:

[Launch UAC Forecasting Dashboard](https://o8neh4rguhcow3wjgvcw8i.streamlit.app/)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Project Features

- Historical data analysis and visualization
- Data cleaning and preprocessing
- Weekly data aggregation and completeness checks
- Lag-based feature engineering
- Comparison of forecasting models
- Four-week forecasting for three demand measures
- Interactive dashboard with date filtering
- Downloadable forecast results in CSV format

## Forecasting Models

Three approaches were evaluated:

1. Random Forest Regression
2. Linear Regression
3. Previous Week Baseline

### Selected Forecasting Methods

| Forecasting Task | Selected Method |
|---|---|
| HHS Care Load | Previous Week Baseline |
| Placement Demand | Previous Week Baseline |
| Transfer Demand | Random Forest |

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Project Structure

```text
UAC-Forecasting-Dashboard/
│
├── app.py
├── weekly_forecasting_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd UAC-Forecasting-Dashboard
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Important Note

The forecasts are estimates based on historical patterns and the last available observation in the dataset. They should not be interpreted as guaranteed outcomes or actual recorded values.

The dataset used in this project contains historical observations through December 2025. Therefore, the displayed four-week forecast represents the period immediately following the dataset's final observation.

## Learning Outcomes

Through this project, I gained practical experience in:

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Regression models and model evaluation
- Time-based forecasting
- Interactive dashboard development
- Communicating analytical results through visualizations

## Author

**Khushi Tadyal**

Computer Science and Engineering Student

---

*This project was developed as a practical predictive analytics project using Python and machine learning techniques.*
