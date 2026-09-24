# Quantity Analysis - Time Series Forecasting

A Python-based time series analysis project for understanding **daily quantity sales trends** and building a baseline **ARIMA forecasting model** using historical transaction data.

The project transforms transaction-level sales records into a daily time series, explores short-term trends using a moving average, decomposes the series into trend/seasonal/residual components, and evaluates an ARIMA model on a chronological holdout set.

---

## Project Overview

Time series forecasting is useful when historical observations are indexed by time and the objective is to understand patterns or estimate future values.

In this project, the target variable is:

> **`total_qty_sales` — total quantity sold**

Because the raw dataset contains multiple records for the same date, the transaction-level quantity values are aggregated by `OrderDate` to create a single **daily quantity series**.

### Objectives

- Load and inspect the historical sales dataset.
- Convert and validate the order-date field.
- Aggregate transaction-level quantities into daily totals.
- Visualize daily sales and a 7-day moving average.
- Decompose the daily series using a weekly seasonal period.
- Split the data chronologically into training and testing sets.
- Train an **ARIMA(2, 1, 2)** baseline forecasting model.
- Evaluate the forecast using **Root Mean Squared Error (RMSE)**.
- Compare forecasted values with actual test observations visually.

---

## Dataset

The project uses the Excel file:

```text
Raw Data_Predictive Analysis.xlsx
```

### Dataset size

- **Rows:** 40,563
- **Columns:** 9
- **Date range:** January 1, 2019 to December 31, 2020
- **Unique calendar dates:** 731

### Dataset columns

| Column | Description | Used in Model? |
|---|---|---:|
| `OrderDate` | Date of the sales/order record | Yes |
| `ParentProductIdNew` | Parent product identifier | No |
| `ParentProductNew` | Parent product name | No |
| `ProductCategoryNew` | Product category | No |
| `ArtistNameNew` | Artist associated with the product | No |
| `total_qty_sales` | Quantity sold in the record | **Yes — Target** |
| `Selling Price` | Selling price associated with the record | No |
| `productListViews` | Product-list views | No |
| `productListClicks` | Product-list clicks | No |

The raw dataset contains missing values in `productListViews` and `productListClicks`. These columns are not required for the current time series model, so they do not affect the forecasting workflow.

---

## Project Structure

```text
Quantity-Analysis-Time-Series-Data/
│
├── Quantity_Analysis_TimeSeries_Completed (1).ipynb
├── Raw Data_Predictive Analysis.xlsx
└── README.md
```

A cleaner notebook filename can also be used:

```text
Quantity_Analysis_TimeSeries.ipynb
```

---

## Technologies and Libraries

- **Python 3.x**
- **Pandas** — data loading, cleaning, aggregation, and time-series operations
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — plotting support
- **Statsmodels** — seasonal decomposition and ARIMA modeling
- **Scikit-learn** — model evaluation with RMSE
- **Jupyter Notebook** — interactive execution and analysis

---

## Methodology

### 1. Load the dataset

The Excel dataset is loaded using Pandas:

```python
df = pd.read_excel('Raw Data_Predictive Analysis.xlsx')
```

The notebook then checks the shape, columns, and initial records.

---

### 2. Convert and clean the date column

`OrderDate` is converted to a Pandas datetime type. The workflow also supports Excel-style numeric serial dates:

```python
if pd.api.types.is_numeric_dtype(df['OrderDate']):
    df['OrderDate'] = pd.to_datetime(
        df['OrderDate'],
        unit='D',
        origin='1899-12-30',
        errors='coerce'
    )
else:
    df['OrderDate'] = pd.to_datetime(
        df['OrderDate'],
        errors='coerce'
    )
```

The target quantity column is converted to numeric values and invalid records are removed.

---

### 3. Create the daily time series

The raw data has multiple sales records per date. Therefore, the target is aggregated by day:

```python
daily_quantity = (
    df.set_index('OrderDate')['total_qty_sales']
      .resample('D')
      .sum()
      .asfreq('D')
      .fillna(0)
)
```

This produces **731 daily observations**, one for each calendar day in the dataset period.

The resulting time series has:

- **Total quantity:** 581,489
- **Average daily quantity:** approximately 795.47
- **Minimum daily quantity:** 91
- **Maximum daily quantity:** 11,424

---

### 4. Moving average analysis

A **7-day moving average** is calculated to smooth short-term fluctuations and make the underlying weekly trend easier to observe.

```python
daily_quantity.rolling(7).mean()
```

The visualization contains both:

- Raw daily quantity
- 7-day moving average

This helps identify changes in the level of sales while reducing day-to-day noise.

---

### 5. Seasonal decomposition

The daily series is decomposed using an additive model with a **7-day seasonal period**:

```python
decomposition = seasonal_decompose(
    ts,
    model='additive',
    period=7,
    extrapolate_trend='freq'
)
```

The decomposition separates the observed series into:

```text
Observed = Trend + Seasonal + Residual
```

The weekly period of 7 was selected because the observations are daily and a weekly cycle is a natural time scale to inspect.

---

### 6. Train-test split

The dataset is split chronologically rather than randomly, which avoids using future observations to train the forecasting model.

The split used in the notebook is:

- **Training:** first 80% = 584 days
- **Testing:** final 20% = 147 days

The corresponding dates are approximately:

```text
Training: 2019-01-01 to 2020-08-06
Testing : 2020-08-07 to 2020-12-31
```

---

### 7. ARIMA modeling

A baseline **ARIMA(2, 1, 2)** model is fitted to the training series:

```python
model = ARIMA(
    train,
    order=(2, 1, 2),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit()
```

The ARIMA parameters mean:

- **p = 2:** two autoregressive terms
- **d = 1:** first-order differencing
- **q = 2:** two moving-average terms

The fitted model is then used to forecast the full 147-day test period.

---

## Evaluation

The forecasting model is evaluated using **Root Mean Squared Error (RMSE)**.

### RMSE formula

```text
RMSE = sqrt(mean((Actual - Forecast)^2))
```

The notebook produced:

```text
RMSE: 1000.85
```

This value is a **baseline result on the final 20% chronological test period**. It should be used as a reference point for comparing improved forecasting approaches rather than interpreted in isolation as a universal measure of model quality.

---

## Visualizations

The notebook generates the following visual outputs:

### 1. Daily Quantity and Moving Average

Shows the complete daily quantity series together with the 7-day moving average.

### 2. Seasonal Decomposition

Displays:

- Observed series
- Trend
- Weekly seasonal component
- Residual component

### 3. ARIMA Forecast vs Actual

Compares:

- Training observations
- Actual test observations
- ARIMA forecast

This makes it easier to visually inspect how the baseline model follows the observed series.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/AYUSHMSINGH2004/Quantity-Analysis-Time-Series-Data.git
cd Quantity-Analysis-Time-Series-Data
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn openpyxl jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Quantity_Analysis_TimeSeries_Completed (1).ipynb
```

Make sure the Excel dataset is in the **same directory** as the notebook.

### 5. Run all cells

Use:

```text
Kernel → Restart & Run All
```

The notebook will load the dataset, perform the time-series analysis, train the ARIMA model, calculate RMSE, and display the forecast plots.

---

## Reproducibility

The analysis is deterministic with respect to the provided dataset and chronological train-test split. No random train-test split is used because time series observations must preserve temporal order.

The key modeling setup is:

```python
train_size = int(len(ts) * 0.80)
train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

ARIMA(train, order=(2, 1, 2))
```

---

## Why ARIMA?

ARIMA is a classical statistical model designed for univariate time series forecasting. It provides a useful baseline because it models temporal dependence using autoregression, differencing, and moving-average components.

For this project, only historical daily quantity is used as the forecasting variable. Other dataset features such as selling price, views, clicks, product, category, and artist are not included in the current ARIMA model.

---

## Limitations

This project is intentionally implemented as a baseline time series forecasting workflow. Several improvements are possible:

- The ARIMA order `(2,1,2)` is a fixed baseline rather than the result of a full model-selection procedure.
- Weekly seasonality is explored during decomposition but is not explicitly modeled in the ARIMA specification.
- The model is univariate and does not use potentially informative external variables such as price, product category, views, or clicks.
- Large spikes in the daily quantity series can increase RMSE substantially.
- The current evaluation uses a single chronological holdout period.

---

## Future Improvements

Potential extensions include:

### SARIMA

Use seasonal ARIMA to explicitly model weekly seasonality:

```text
SARIMA(p,d,q)(P,D,Q,7)
```

### Prophet

Prophet can be evaluated for trend and seasonal components, especially when additional calendar effects are important.

### Machine Learning Models

Create lag, rolling, calendar, price, views, and clicks features and compare models such as:

- Random Forest Regressor
- XGBoost
- LightGBM

### Deep Learning

Sequence models such as:

- LSTM
- GRU
- Temporal Convolutional Networks

can be tested for longer and more complex temporal relationships.

### Better Model Selection

Compare multiple candidate models using metrics such as:

- RMSE
- MAE
- MAPE

and use rolling or walk-forward validation for a more robust assessment.

---

## Key Takeaways

- The transaction-level data was successfully converted into a daily quantity time series.
- The dataset spans **731 calendar days** from 2019 through 2020.
- A 7-day moving average was used to smooth short-term variation.
- Seasonal decomposition was performed with a weekly period of 7.
- An **ARIMA(2,1,2)** model was used as the forecasting baseline.
- The model achieved an **RMSE of 1000.85** on the final 147-day test period.
- The project provides a foundation for testing seasonal, machine-learning, and deep-learning forecasting methods.

---

## Author

**Ayush M Singh**

GitHub: [AYUSHMSINGH2004](https://github.com/AYUSHMSINGH2004)

---

## License

This repository is intended for educational and analytical purposes. Add an explicit open-source license such as the MIT License if you would like others to reuse and modify the project under defined terms.
