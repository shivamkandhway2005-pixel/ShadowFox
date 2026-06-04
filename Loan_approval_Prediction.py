# Loan Approval Prediction using Random Forest Classifier

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("loan_prediction_columns_arranged.csv")

# Data Preprocessing

# Remove Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# Fill Missing Values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Create Total Income Feature
df["Total_Income"] = (
    df["Applicant_Income"] +
    df["Coapplicant_Income"]
)

# Log Transformation
df["Loan_Amount"] = np.log(df["Loan_Amount"])
df["Total_Income"] = np.log(df["Total_Income"])

# Encode Categorical Columns
le = LabelEncoder()
categorical_columns = [
    "Gender",
    "Marital_Status",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
# Feature Selection
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# Random Forest Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ",accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
sample = pd.DataFrame({
    "Gender": [1],                # Male
    "Marital_Status": [1],        # Married
    "Dependents": [1],
    "Education": [0],             # Graduate
    "Self_Employed": [0],         # No
    "Applicant_Income": [5000],
    "Coapplicant_Income": [2000],
    "Loan_Amount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1],
    "Property_Area": [2],         # Urban
    "Total_Income": [7000]
})

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Rejected")