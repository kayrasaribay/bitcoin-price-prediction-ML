Aşağıdaki README’yi doğrudan `README.md` içine koyabilirsin. İçerik, yüklediğin notebook/proje akışına göre hazırlandı.

````markdown
# Bitcoin Price Prediction Using Machine Learning

A machine learning regression project focused on short-term Bitcoin price prediction using historical 1-minute OHLCV market data.

This project covers the full machine learning workflow:

- Data loading and initial inspection
- Exploratory data analysis
- Data cleaning
- Feature engineering
- Regression model training
- Model evaluation
- Final prediction and error analysis

The main goal is to predict the **next 1-minute Bitcoin closing price** using historical market features and engineered indicators.

---

## Project Overview

Bitcoin prices are highly volatile and difficult to predict, especially in short time intervals. This project explores whether machine learning regression models can capture short-term price behavior using historical Bitcoin market data.

The dataset contains historical Bitcoin market data recorded at 1-minute intervals. The project uses price, volume, volatility, return-based features, moving averages, and lag features to predict the next closing price.

This is a **regression problem**, where the target variable is:

```text
Next_Close
````

The project should be interpreted as a **short-horizon price prediction system**, not a long-term Bitcoin forecasting or investment decision tool.

---

## Project Objective

The main objectives of this project are:

* Analyze historical Bitcoin price behavior
* Understand price trends, volatility, volume, and return distributions
* Build meaningful features from raw OHLCV data
* Compare multiple regression models
* Select the best-performing model
* Evaluate prediction performance using regression metrics
* Demonstrate an end-to-end machine learning workflow

---

## Dataset

The dataset consists of historical Bitcoin market data recorded at **1-minute intervals**.

Original dataset columns:

| Column      | Description                               |
| ----------- | ----------------------------------------- |
| `Timestamp` | Unix timestamp value                      |
| `Open`      | Opening price of Bitcoin for the interval |
| `High`      | Highest price during the interval         |
| `Low`       | Lowest price during the interval          |
| `Close`     | Closing price of Bitcoin for the interval |
| `Volume`    | Trading volume during the interval        |

Additional columns were created during preprocessing and feature engineering.

Target variable:

| Target       | Description                                 |
| ------------ | ------------------------------------------- |
| `Next_Close` | Closing price of the next 1-minute interval |

Dataset size:

```text
Original dataset: ~7.5 million rows
Frequency: 1-minute market records
```

---

## Project Structure

```bash
bitcoin-price-prediction-ML/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_loading_and_overview.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning_and_feature_engineering.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_final_prediction_and_conclusion.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── visualization/
│   └── models/
│
├── models/
│   └── saved regression models
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Notebook Workflow

### 1. Data Loading and Overview

The first notebook focuses on understanding the raw dataset.

Main steps:

* Loaded the Bitcoin historical dataset
* Inspected dataset shape, columns, and data types
* Checked missing values
* Checked duplicated timestamps
* Converted Unix timestamp values into readable datetime format
* Created initial visualizations for price and volume behavior

Key observations:

* The dataset contains high-frequency Bitcoin market data.
* No missing values were found in the original OHLCV columns.
* Timestamp values were successfully converted into datetime format.
* The dataset is suitable for time-series-based regression modeling.

---

### 2. Exploratory Data Analysis

The second notebook explores Bitcoin market behavior using visualizations and statistical analysis.

Main EDA topics:

* Bitcoin closing price trend
* Trading volume over time
* Price distribution
* Return distribution
* Volatility analysis
* Moving averages
* Correlation analysis
* Monthly volatility and volume patterns
* Actual market behavior interpretation

Key findings:

* Bitcoin prices show strong long-term growth.
* The price distribution is heavily right-skewed.
* Most short-term returns are concentrated around zero.
* OHLC features are highly correlated.
* Volatility changes significantly over time.
* Monthly seasonality is not strongly visible in volume or volatility.

---

### 3. Data Cleaning and Feature Engineering

The third notebook prepares the dataset for machine learning.

Created features include:

| Feature              | Description                           |
| -------------------- | ------------------------------------- |
| `Volatility`         | `High - Low`                          |
| `Return`             | `(Close - Open) / Open`               |
| `Log_Return`         | `log(Close / Open)`                   |
| `MA_60`              | 60-minute moving average              |
| `MA_240`             | 240-minute moving average             |
| `MA_1440`            | 1440-minute moving average            |
| `Rolling_Volatility` | Rolling standard deviation of returns |
| `Close_Lag_1`        | Previous close price                  |
| `Close_Lag_5`        | Close price 5 minutes ago             |
| `Close_Lag_15`       | Close price 15 minutes ago            |
| `Return_Lag_1`       | Previous return                       |
| `Volume_Lag_1`       | Previous volume                       |
| `Year`               | Year extracted from date              |
| `Month`              | Month extracted from date             |
| `Hour`               | Hour extracted from date              |
| `DayOfWeek`          | Day of week extracted from date       |

Removed or excluded columns:

| Column              | Reason                                                      |
| ------------------- | ----------------------------------------------------------- |
| `Timestamp`         | Replaced by extracted time features                         |
| `Date`              | Raw datetime is not directly used in regression             |
| `Range`             | Duplicated information already represented by `Volatility`  |
| `Cumulative_Return` | Useful for EDA but not selected for final modeling          |
| `Price_change`      | Removed because `Return` provides normalized price movement |

The final model-ready dataset was saved for model training.

---

## Machine Learning Methodology

The modeling task is to predict:

```text
Next_Close
```

using current and historical Bitcoin market features.

A time-based train-test split was used:

```text
Train set: first 80% of observations
Test set: final 20% of observations
```

This approach preserves chronological order and helps avoid data leakage.

### Scaling

Linear models are sensitive to feature scale, so `StandardScaler` was used inside scikit-learn pipelines.

Pipeline structure:

```text
StandardScaler → Regression Model
```

This ensures that scaling is fit only on the training data and then applied to the test data.

---

## Models Used

The following regression models were trained and compared:

| Model                 | Description                              |
| --------------------- | ---------------------------------------- |
| Linear Regression     | Baseline linear regression model         |
| Ridge Regression      | Linear regression with L2 regularization |
| Lasso Regression      | Linear regression with L1 regularization |
| ElasticNet Regression | Combination of L1 and L2 regularization  |

Tree-based models such as Random Forest and Gradient Boosting were considered, but they were excluded from the final comparison due to computational cost on the full high-frequency dataset.

---

## Model Evaluation

The models were evaluated using:

| Metric   | Meaning                                              |
| -------- | ---------------------------------------------------- |
| MAE      | Average absolute prediction error                    |
| MSE      | Average squared prediction error                     |
| RMSE     | Root mean squared error, penalizes large errors more |
| R² Score | Proportion of variance explained by the model        |

### Model Results

| Model                 |     R² Score |     MAE |     RMSE |
| --------------------- | -----------: | ------: | -------: |
| Linear Regression     | 0.9999966771 | 29.3988 |  50.1903 |
| Ridge Regression      | 0.9999966294 | 29.6299 |  50.5493 |
| Lasso Regression      | 0.9999941272 | 39.2567 |  66.7243 |
| ElasticNet Regression | 0.9999731925 | 89.1215 | 142.5571 |

### Best Model

The best-performing model was:

```text
Linear Regression
```

It achieved the highest R² score and the lowest MAE and RMSE values.

However, the very high R² score should be interpreted carefully. Since the target is the next 1-minute closing price, consecutive Bitcoin prices are usually very close to each other. Therefore, this task is easier than long-term forecasting.

---

## Final Prediction

The final notebook loads the saved best model using `joblib` and generates predictions without retraining.

Final prediction workflow:

1. Load the model-ready dataset
2. Load the saved Linear Regression model
3. Recreate the same time-based test split
4. Generate final predictions
5. Compare actual and predicted prices
6. Analyze residuals and percentage errors

The actual vs predicted plot shows that predicted values closely follow actual Bitcoin prices in the short term.

---

## Prediction Error Analysis

Prediction errors were analyzed using:

* Residuals
* Absolute error
* Percentage error
* Actual vs predicted comparison
* Sample prediction table

Residuals were generally small, but larger errors may occur during sudden market movements or high-volatility periods.

This is expected in financial data, especially with high-frequency cryptocurrency prices.

---

## Key Insights

* Bitcoin prices show strong long-term growth but high volatility.
* Most 1-minute returns are close to zero.
* OHLC features are highly correlated.
* Moving averages and lag features are useful for short-term prediction.
* Linear Regression performed best among the tested models.
* High R² scores are expected due to the short prediction horizon.
* The model should not be interpreted as a long-term Bitcoin forecasting system.

---

## Limitations

This project has several important limitations:

* The model predicts only the next 1-minute closing price.
* It does not include news, sentiment, macroeconomic data, or regulatory events.
* Bitcoin is highly volatile and sudden price movements are difficult to predict.
* High model performance is partly due to the short prediction horizon.
* Walk-forward validation or time-series cross-validation could improve robustness.
* The model is not designed for financial advice or investment decisions.

---

## Future Work

Possible improvements:

* Predict longer horizons such as 5-minute, 1-hour, or 1-day ahead prices
* Predict returns instead of raw prices
* Predict price direction as a classification problem
* Try advanced models such as XGBoost, LightGBM, Random Forest, or Gradient Boosting on optimized samples
* Use deep learning models such as LSTM, GRU, or Transformers
* Add external data such as:

  * News sentiment
  * Social media sentiment
  * Macroeconomic indicators
  * Crypto market indexes
  * Funding rates
* Apply walk-forward validation
* Build a dashboard for prediction visualization

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/kayrasaribay/bitcoin-price-prediction-ML.git
```

2. Navigate into the project folder:

```bash
cd bitcoin-price-prediction-ML
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

5. Run notebooks in order:

```text
01_data_loading_and_overview.ipynb
02_exploratory_data_analysis.ipynb
03_data_cleaning_and_feature_engineering.ipynb
04_model_training_and_evaluation.ipynb
05_final_prediction_and_conclusion.ipynb
```

---

## Disclaimer

This project is for educational and portfolio purposes only.

It should not be used as financial advice or as a trading system. Cryptocurrency markets are highly volatile, and machine learning models trained on historical data cannot guarantee future performance.

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for short-term Bitcoin price prediction.

The workflow includes data inspection, exploratory data analysis, feature engineering, model training, model evaluation, final prediction, and error analysis.

The final Linear Regression model achieved strong short-term prediction performance. However, the model should be understood as a next-step forecasting model rather than a long-term Bitcoin price prediction system.

```
```
