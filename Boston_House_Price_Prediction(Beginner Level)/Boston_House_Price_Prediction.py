import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("HousingData.csv")

# Display first 5 rows
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Handle Missing Values
for col in df.columns:
    df[col] = df[col].fillna(df[col].median())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# Feature Importance
feature_names = df.drop("MEDV", axis=1).columns
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# Plot Feature Importance
plt.figure(figsize=(10,6))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()
sample = pd.DataFrame({
    "CRIM": [0.05],      # Crime rate
    "ZN": [20],          # Residential land zoned
    "INDUS": [5.0],      # Non-retail business acres
    "CHAS": [0],         # Charles River dummy variable
    "NOX": [0.45],       # Nitric oxide concentration
    "RM": [6.5],         # Average rooms per dwelling
    "AGE": [45],         # % owner-occupied units built before 1940
    "DIS": [5.0],        # Distance to employment centers
    "RAD": [4],          # Accessibility to highways
    "TAX": [300],        # Property tax rate
    "PTRATIO": [16.0],   # Pupil-teacher ratio
    "B": [390.0],        # Proportion related to Black population index
    "LSTAT": [8.0]       # % lower status population
})
predicted_price = model.predict(sample)
print("Predicted House Price:", predicted_price[0])