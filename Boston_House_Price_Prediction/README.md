# 🏠 Boston House Price Prediction Using Random Forest Regression

## 📌 Overview

Boston House Price Prediction is a Machine Learning regression project developed as part of the **ShadowFox Data Science & Machine Learning Internship**.

The objective of this project is to predict the median value of owner-occupied homes in Boston based on various housing, economic, and demographic features.

A **Random Forest Regressor** is used to learn patterns from historical housing data and estimate property prices with high accuracy.

---

## 🎯 Problem Statement

Accurate house price prediction is an important task in the real estate industry. Property values are influenced by multiple factors such as crime rate, number of rooms, tax rate, accessibility, and neighborhood characteristics.

The goal of this project is to build a Machine Learning model capable of predicting house prices based on these features and providing insights into the factors that influence property values.

---

## 📊 Dataset Information

The dataset contains information collected from different suburbs of Boston.

### Features

| Feature | Description                                                    |
| ------- | -------------------------------------------------------------- |
| CRIM    | Per capita crime rate by town                                  |
| ZN      | Proportion of residential land zoned for large lots            |
| INDUS   | Proportion of non-retail business acres per town               |
| CHAS    | Charles River dummy variable (1 if tract bounds river, else 0) |
| NOX     | Nitric oxide concentration                                     |
| RM      | Average number of rooms per dwelling                           |
| AGE     | Proportion of owner-occupied units built before 1940           |
| DIS     | Weighted distance to employment centers                        |
| RAD     | Accessibility to radial highways                               |
| TAX     | Property tax rate                                              |
| PTRATIO | Pupil-teacher ratio                                            |
| B       | Proportion related to Black population index                   |
| LSTAT   | Percentage of lower status population                          |
| MEDV    | Median value of owner-occupied homes (Target Variable)         |

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection

* Load the dataset using Pandas.
* Explore the dataset structure.
* Identify missing values.

### 2️⃣ Data Preprocessing

* Handle missing values using median imputation.
* Separate features and target variable.

### 3️⃣ Train-Test Split

The dataset is divided into:

* Training Data → 80%
* Testing Data → 20%

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 4️⃣ Model Training

A Random Forest Regressor is trained on the dataset.

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

### 5️⃣ Prediction

The trained model predicts house prices for unseen data.

### 6️⃣ Model Evaluation

Performance is measured using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### 7️⃣ Feature Importance Analysis

Random Forest feature importance is used to identify the most influential factors affecting house prices.

---

## 🤖 Machine Learning Model

### Random Forest Regressor

Random Forest Regression is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### Advantages

✅ High Prediction Accuracy

✅ Handles Non-Linear Relationships

✅ Reduces Overfitting

✅ Robust Against Noise

✅ Provides Feature Importance

---

## 📈 Model Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

Provides error magnitude in the same unit as house prices.

### R² Score

Measures how well the model explains variability in house prices.

---

## 📊 Feature Importance

The model calculates feature importance scores to determine which variables have the greatest impact on house prices.

Commonly important features include:

* RM (Average Number of Rooms)
* LSTAT (Lower Status Population Percentage)
* PTRATIO (Pupil-Teacher Ratio)
* TAX (Property Tax Rate)
* CRIM (Crime Rate)

A horizontal bar chart is generated to visualize feature importance.

---

## 🧪 Sample Prediction

Example Input:

```python
sample = pd.DataFrame({
    "CRIM": [0.05],
    "ZN": [20],
    "INDUS": [5.0],
    "CHAS": [0],
    "NOX": [0.45],
    "RM": [6.5],
    "AGE": [45],
    "DIS": [5.0],
    "RAD": [4],
    "TAX": [300],
    "PTRATIO": [16.0],
    "B": [390.0],
    "LSTAT": [8.0]
})
```

Example Output:

```text
Predicted House Price: 22.83
```

*(Output may vary depending on training results.)*

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Jupyter Notebook

---

## 📂 Project Structure

```text
Boston-House-Price-Prediction/
│
├── HousingData.csv
├── Boston_House_Price_Prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/shivamkandhway2005-pixel/ShadowFox.git
```

### Navigate to Project Directory

```bash
cd ShadowFox
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the Python script:

```bash
python boston_house_price_prediction.py
```

Or launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📊 Key Insights

* The number of rooms (`RM`) strongly influences house prices.
* Crime rate (`CRIM`) negatively impacts property values.
* Socioeconomic indicators such as `LSTAT` significantly affect housing prices.
* Random Forest Regression effectively captures complex relationships between housing features and prices.

---

## 🔮 Future Improvements

* Hyperparameter Tuning using GridSearchCV
* Model Comparison with:

  * Linear Regression
  * Decision Tree Regressor
  * XGBoost Regressor
  * Gradient Boosting Regressor
* Streamlit Web Application Deployment
* Advanced Feature Engineering

---

## 🎓 Skills Demonstrated

* Data Cleaning & Preprocessing
* Regression Modeling
* Feature Importance Analysis
* Model Evaluation
* Data Visualization
* Machine Learning Workflow
* Git & GitHub

---

## 🏆 ShadowFox Internship Project

This project was completed as part of the **ShadowFox Data Science & Machine Learning Internship Program**, demonstrating the application of Machine Learning techniques to solve real-world regression problems.

---

## 👨‍💻 Author

**Shivam Kumar**

🔗 Repository: https://github.com/shivamkandhway2005-pixel/ShadowFox

🔗 GitHub Profile: https://github.com/shivamkandhway2005-pixel

---

⭐ If you found this project useful, consider giving the repository a star.
