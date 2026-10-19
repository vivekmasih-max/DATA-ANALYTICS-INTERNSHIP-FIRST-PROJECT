import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

data = joblib.load("nassau_preprocessed.pkl")

# Processed data for ML
X_train = data["X_train"]
X_test = data["X_test"]

# Target
y_train = data["y_train"]
y_test = data["y_test"]

# Original unprocessed test rows
X_test_original = data["X_test_original"]


# ============================================================
# 2. CHECK DATA
# ============================================================

print("Processed X_train:", X_train.shape)
print("Processed X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nOriginal X_test:")
print(X_test_original.head())


# ============================================================
# 3. TRAIN LINEAR REGRESSION
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ============================================================
# 4. PREDICT
# ============================================================

y_pred = model.predict(X_test)

print("\nPrediction successful!")
print("First 10 predictions:")
print(y_pred[:10])


# ============================================================
# 5. CREATE PREDICTION REPORT
# ============================================================

results = X_test_original.copy()

results["Actual_Lead_Time"] = y_test.values

results["Predicted_Lead_Time"] = y_pred

results["Error"] = (
    results["Predicted_Lead_Time"]
    - results["Actual_Lead_Time"]
)

results["Absolute_Error"] = abs(
    results["Error"]
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\nPrediction Results:\n")

print(
    results[
        [
            "Product Name",
            "Factory",
            "Region",
            "Actual_Lead_Time",
            "Predicted_Lead_Time",
            "Error",
            "Absolute_Error"
        ]
    ].head(10000)
)
# ============================================================
# 7. MODEL EVALUATION
# ============================================================
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n===================================")
print("LINEAR REGRESSION MODEL EVALUATION")
print("===================================")

print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

results.to_csv(
    "Linear_Regression_Lead_Time_Predictions.csv",
    index=False
)

print("\nPrediction report saved successfully.")