# Boston House Price Prediction using Random Forest Regression

## Project Overview

This project aims to predict housing prices in Boston using Machine Learning. A Random Forest Regressor is used to learn the relationship between various housing features such as crime rate, number of rooms, tax rate, and population characteristics, and then predict the median value of owner-occupied homes.

---

## Problem Statement

The objective of this project is to build a regression model capable of accurately predicting Boston house prices based on housing and neighborhood characteristics. The project involves data preprocessing, handling missing values, model training, evaluation, and feature importance analysis.

---

## Dataset Information

The dataset contains information about housing in Boston suburbs.

### Features

* CRIM – Per capita crime rate by town
* ZN – Proportion of residential land zoned for large lots
* INDUS – Proportion of non-retail business acres per town
* CHAS – Charles River dummy variable
* NOX – Nitric oxide concentration
* RM – Average number of rooms per dwelling
* AGE – Proportion of owner-occupied units built before 1940
* DIS – Weighted distance to employment centers
* RAD – Accessibility to radial highways
* TAX – Property tax rate
* PTRATIO – Pupil-teacher ratio
* B – Proportion related to population demographics
* LSTAT – Percentage of lower-status population

### Target Variable

* MEDV – Median value of owner-occupied homes (in $1000s)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

---

## Machine Learning Algorithm

### Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Model Parameters:

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

---

## Data Preprocessing

1. Loaded the dataset using Pandas.
2. Checked for missing values.
3. Replaced missing values using the median of each column.
4. Split the dataset into training and testing sets (80:20).

---

## Model Evaluation Metrics

The following metrics were used:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Results

| Metric   | Value |
| -------- | ----- |
| MAE      | 2.02  |
| MSE      | 8.61  |
| RMSE     | 2.93  |
| R² Score | 0.88  |

---

## Feature Importance

The Random Forest model was used to determine the importance of each feature in predicting house prices.

Important features typically include:

* RM (Average Rooms)
* LSTAT (Lower Status Population)
* CRIM (Crime Rate)
* PTRATIO (Pupil-Teacher Ratio)

---

## Visualization

The project includes a feature importance graph generated using Matplotlib to visualize the contribution of each feature.

---

## Project Structure

```text
Boston-House-Price-Prediction/
│
├── HousingData.csv
├── Boston_House_Price_Prediction.ipynb
├── README.md
├── requirements.txt
└── output.png
```

---

## How to Run

1. Clone the repository

```bash
git clone <repository-link>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook

```bash
jupyter notebook
```

---

## Conclusion

A Random Forest Regression model was successfully developed to predict Boston house prices. The model achieved an R² score of 0.88, indicating strong predictive performance. The results demonstrate that ensemble learning techniques such as Random Forest can effectively capture complex relationships in housing data and provide accurate price predictions.

---

## Author

Shivam Kumar

ShadowFox Machine Learning Internship Project
