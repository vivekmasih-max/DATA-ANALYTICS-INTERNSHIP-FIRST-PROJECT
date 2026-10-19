import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    'C:/SEMESTER4/PROJECTWORK/CANDYDATA.csv'
)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

numerical_features = [
    "Units",
    "Sales",
    "Cost",
    "Distance_km(F TO C)"
]

categorical_features = [
    "Region",
    "Ship Mode",
    "Division",
    "Product Name",
    "Factory"
]

target = "Lead_Time(Yrs)"


X = df[numerical_features + categorical_features]

y = df[target]


# --------------------------------------------------
# 3. Split BEFORE preprocessing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 4. Separate numerical and categorical data
# --------------------------------------------------

X_train_numeric = X_train[numerical_features]
X_test_numeric = X_test[numerical_features]

X_train_categorical = X_train[categorical_features]
X_test_categorical = X_test[categorical_features]


# --------------------------------------------------
# 5. Normalize numerical features
# --------------------------------------------------

scaler = StandardScaler()

X_train_numeric_scaled = scaler.fit_transform(
    X_train_numeric
)

X_test_numeric_scaled = scaler.transform(
    X_test_numeric
)


# --------------------------------------------------
# 6. Encode categorical variables
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_categorical_encoded = encoder.fit_transform(
    X_train_categorical
)

X_test_categorical_encoded = encoder.transform(
    X_test_categorical
)


# --------------------------------------------------
# 7. Create training-ready feature matrix
# --------------------------------------------------

import numpy as np

X_train_ready = np.hstack([
    X_train_numeric_scaled,
    X_train_categorical_encoded
])

X_test_ready = np.hstack([
    X_test_numeric_scaled,
    X_test_categorical_encoded
])


# --------------------------------------------------
# 8. Check dimensions
# --------------------------------------------------

#print("X_train shape:", X_train_ready.shape)
#print("X_test shape:", X_test_ready.shape)

#print("y_train shape:", y_train.shape)
#print("y_test shape:", y_test.shape)



import joblib

preprocessed_data = {
    # Processed data used by ML model
    "X_train": X_train_ready,
    "X_test": X_test_ready,

    # Target values
    "y_train": y_train,
    "y_test": y_test,

    # ORIGINAL rows - important for interpreting predictions
    "X_train_original": X_train,
    "X_test_original": X_test,

    # Preprocessing objects
    "scaler": scaler,
    "encoder": encoder
}

joblib.dump(
    preprocessed_data,
    "nassau_preprocessed.pkl"
)

print("All preprocessing results saved.")