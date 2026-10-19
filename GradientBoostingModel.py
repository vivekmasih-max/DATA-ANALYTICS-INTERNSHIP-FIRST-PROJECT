import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

data = joblib.load("nassau_preprocessed.pkl")

# Processed data for ML
X_train = data["X_train"]
X_test = data["X_test"]

# Target values
y_train = data["y_train"]
y_test = data["y_test"]

# Original test rows
X_test_original = data["X_test_original"]


# ============================================================
# 2. CHECK DATA
# ============================================================

print("Processed X_train:", X_train.shape)
print("Processed X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nOriginal X_test shape:", X_test_original.shape)


# ============================================================
# 3. CREATE GRADIENT BOOSTING REGRESSOR
# ============================================================

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nGradient Boosting training completed!")


# ============================================================
# 5. PREDICT LEAD TIME
# ============================================================

y_pred = model.predict(X_test)

print("\nPrediction successful!")

print("\nFirst 10 predictions:")
print(y_pred[:10])


# ============================================================
# 6. CREATE PREDICTION RESULTS TABLE
# ============================================================

results = X_test_original.copy()

# Actual Lead Time
results["Actual_Lead_Time"] = y_test.values

# Predicted Lead Time
results["Predicted_Lead_Time"] = y_pred

# Prediction error
results["Error"] = (
    results["Predicted_Lead_Time"]
    - results["Actual_Lead_Time"]
)

# Absolute error
results["Absolute_Error"] = abs(
    results["Error"]
)


# ============================================================
# 7. DISPLAY FIRST 10 PREDICTIONS
# ============================================================

print("\nPrediction Results:\n")

print(
    results[
        [
            "Product Name",
            "Factory",
            "Region",
            "Ship Mode",
            "Actual_Lead_Time",
            "Predicted_Lead_Time",
            "Error",
            "Absolute_Error"
        ]
    ].head(10)
)


# ============================================================
# 8. MODEL EVALUATION
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


#print("\n===================================")
print("GRADIENT BOOSTING MODEL EVALUATION")
print("===================================")

print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)


# ============================================================
# 9. SAVE PREDICTION RESULTS
# ============================================================

#results.to_csv(
    #"Gradient_Boosting_Lead_Time_Predictions.csv",
    #index=False
#)

#print("\nPrediction report saved successfully!")