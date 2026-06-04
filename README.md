# 🏦 Loan Approval Prediction Using Random Forest Classifier

## 📌 Overview

Loan Approval Prediction is a Machine Learning classification project developed as part of the **ShadowFox Data Science & Machine Learning Internship**.

The objective of this project is to predict whether a loan application will be **Approved** or **Rejected** based on an applicant's demographic and financial information.

A **Random Forest Classifier** is used to analyze historical loan application data and make intelligent approval decisions. This helps financial institutions automate the loan evaluation process and improve decision-making efficiency.

---

## 🎯 Problem Statement

Loan approval is a critical process in the banking and fintech sector. Manually reviewing loan applications can be time-consuming and may lead to inconsistencies.

The goal of this project is to build a Machine Learning model capable of predicting loan approval status using applicant information such as:

* Income
* Credit History
* Education
* Employment Status
* Loan Amount
* Property Area
* Marital Status

---

## 📊 Dataset Information

The dataset contains historical records of loan applicants along with their approval status.

### Features

| Feature            | Description                           |
| ------------------ | ------------------------------------- |
| Loan_ID            | Unique Loan Identifier                |
| Gender             | Applicant Gender                      |
| Marital_Status     | Marital Status                        |
| Dependents         | Number of Dependents                  |
| Education          | Education Level                       |
| Self_Employed      | Self-Employment Status                |
| Applicant_Income   | Applicant Income                      |
| Coapplicant_Income | Co-Applicant Income                   |
| Loan_Amount        | Requested Loan Amount                 |
| Loan_Amount_Term   | Loan Repayment Duration               |
| Credit_History     | Credit History Record                 |
| Property_Area      | Property Location                     |
| Loan_Status        | Target Variable (Approved / Rejected) |

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection

* Load dataset using Pandas.
* Inspect data structure and missing values.

### 2️⃣ Data Preprocessing

#### Remove Unnecessary Features

The `Loan_ID` column is removed since it does not contribute to prediction.

#### Handle Missing Values

* Mode is used for categorical features.
* Median is used for numerical features.

### 3️⃣ Feature Engineering

A new feature called **Total_Income** is created:

```python
df["Total_Income"] = (
    df["Applicant_Income"] +
    df["Coapplicant_Income"]
)
```

This provides a better representation of the applicant's financial strength.

### 4️⃣ Log Transformation

Applied to:

* Loan_Amount
* Total_Income

Benefits:

* Reduces skewness
* Handles extreme values
* Improves data distribution

```python
df["Loan_Amount"] = np.log(df["Loan_Amount"])
df["Total_Income"] = np.log(df["Total_Income"])
```

### 5️⃣ Label Encoding

Categorical variables are converted into numerical form using **LabelEncoder**.

### 6️⃣ Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

Stratified sampling is used to maintain class balance.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## 🤖 Machine Learning Model

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
```

### Advantages

✅ High Accuracy

✅ Reduces Overfitting

✅ Handles Missing and Noisy Data

✅ Suitable for Classification Problems

✅ Provides Stable Predictions

---

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy Score

Measures the percentage of correct predictions.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

Provides:

* Precision
* Recall
* F1-Score

---

## 🧪 Sample Prediction

The model can predict loan approval for new applicants.

### Example Input

```python
{
    "Gender": 1,
    "Marital_Status": 1,
    "Dependents": 1,
    "Education": 0,
    "Self_Employed": 0,
    "Applicant_Income": 5000,
    "Coapplicant_Income": 2000,
    "Loan_Amount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 2,
    "Total_Income": 7000
}
```

### Example Output

```text
Loan Approved
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Jupyter Notebook

---

## 📂 Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_prediction_columns_arranged.csv
├── Loan_Approval_Prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/shivamkandhway2005-pixel/ShadowFox.git
```

### Navigate to Project Folder

```bash
cd ShadowFox
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the Python script:

```bash
python loan_approval_prediction.py
```

Or launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📊 Key Insights

* Credit History plays the most important role in loan approval decisions.
* Total Income significantly impacts eligibility.
* Feature engineering improves model performance.
* Random Forest provides reliable classification results for this dataset.

---

## 🔮 Future Improvements

* Hyperparameter Tuning using GridSearchCV
* Feature Selection Optimization
* Streamlit Dashboard Deployment
* Flask REST API Development
* Comparison with:

  * Logistic Regression
  * Decision Tree
  * XGBoost
  * Support Vector Machine (SVM)

---

## 🎓 Skills Demonstrated

* Data Cleaning & Preprocessing
* Feature Engineering
* Classification Modeling
* Model Evaluation
* Financial Data Analysis
* Machine Learning Workflow
* Git & GitHub

---

## 🏆 ShadowFox Internship Project

This project was completed as part of the **ShadowFox Data Science & Machine Learning Internship Program**, demonstrating the practical application of Machine Learning techniques to solve real-world financial classification problems.

---

## 👨‍💻 Author

**Shivam Kumar**

🔗 Repository: https://github.com/shivamkandhway2005-pixel/ShadowFox

🔗 GitHub Profile: https://github.com/shivamkandhway2005-pixel

---

⭐ If you found this project useful, consider giving the repository a star.
