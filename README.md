# Boston House Price Prediction using Random Forest Regression

## Project Overview

This project predicts Boston house prices using Machine Learning. A Random Forest Regressor is trained on housing features such as crime rate, number of rooms, tax rate, and population characteristics to estimate house prices accurately.

---

## Problem Statement

Develop a regression model capable of predicting Boston house prices based on housing and neighborhood characteristics. The project includes data preprocessing, missing value handling, model training, evaluation, and feature importance analysis.

---

## Dataset Information

The Boston Housing dataset contains information about housing areas in Boston suburbs.

### Input Features

* CRIM – Per capita crime rate by town
* ZN – Proportion of residential land zoned for large lots
* INDUS – Proportion of non-retail business acres per town
* CHAS – Charles River dummy variable
* NOX – Nitric oxide concentration
* RM – Average number of rooms per dwelling
* AGE – Proportion of owner-occupied units built before 1940
* DIS – Weighted distances to employment centers
* RAD – Accessibility to radial highways
* TAX – Property tax rate
* PTRATIO – Pupil-teacher ratio
* B – Population demographic factor
* LSTAT – Percentage of lower-status population

### Target Variable

* MEDV – Median value of owner-occupied homes (in $1000s)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## Machine Learning Model

### Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

Random Forest combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Data Preprocessing

1. Loaded dataset using Pandas.
2. Checked for missing values.
3. Replaced missing values using column medians.
4. Split dataset into training and testing sets (80:20).

---

## Model Performance

| Metric   | Value |
| -------- | ----- |
| MAE      | 2.02  |
| MSE      | 8.61  |
| RMSE     | 2.93  |
| R² Score | 0.88  |

### Performance Analysis

* The model explains approximately 88% of the variance in house prices.
* Low MAE and RMSE indicate accurate predictions.
* Random Forest performed significantly better than Linear Regression on this dataset.

---

## Feature Importance

Top features influencing house prices:

1. RM (Average Number of Rooms)
2. LSTAT (Lower Status Population)
3. DIS (Distance to Employment Centers)
4. CRIM (Crime Rate)

---

## Visualizations

### Feature Importance

Displays the contribution of each feature toward house price prediction.

### Actual vs Predicted Prices

Compares actual house prices with model predictions.

---

## Project Structure

```text
Boston_House_Price_Prediction/
│
├── Boston_House_Price_Prediction.py
├── HousingData.csv
├── README.md
│
└── images/
    ├── feature_importance.png
    └── actual_vs_predicted.png
```

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install required libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Run the Python file

```bash
python Boston_House_Price_Prediction.py
```

---

## Conclusion

A Random Forest Regression model was successfully developed to predict Boston house prices. The model achieved an R² score of 0.88, demonstrating strong predictive performance. The analysis showed that the number of rooms and socioeconomic factors are the most influential variables affecting housing prices.

---

## Author

**Shivam Kumar**

ShadowFox Machine Learning Internship Project
