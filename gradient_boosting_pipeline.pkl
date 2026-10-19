import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "Nassau_Candy_Distributor.csv"
)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "Product Name",
    "Factory",
    "Region",
    "Ship Mode",
    "Distance_km(F TO C)",
    "Units",
    "Sales",
    "Cost",
    "Division"
]

target = "Lead_Time(Yrs)"


X = df[features].copy()
y = df[target].copy()


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. CATEGORICAL / NUMERICAL COLUMNS
# ============================================================

categorical_features = [
    "Product Name",
    "Factory",
    "Region",
    "Ship Mode",
    "Division"
]

numerical_features = [
    "Distance_km(F TO C)",
    "Units",
    "Sales",
    "Cost"
]


# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 6. GRADIENT BOOSTING
# ============================================================

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


# ============================================================
# 7. COMPLETE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", gb_model)
    ]
)


# ============================================================
# 8. TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 9. SAVE COMPLETE MODEL
# ============================================================

joblib.dump(
    model,
    "gradient_boosting_pipeline.pkl"
)


print("Gradient Boosting pipeline saved successfully.")
