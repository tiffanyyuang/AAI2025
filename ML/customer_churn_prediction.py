# Customer Churn Prediction - Logistic Regression
# Data Source (replace with a real dataset when ready):
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Create a larger sample dataset (120 records)
# -----------------------------
np.random.seed(42)
n = 120

regions = np.random.choice(["West", "East", "South", "North"], size=n, p=[0.3, 0.25, 0.25, 0.2])
age = np.random.randint(18, 70, size=n)
monthly_usage = np.random.randint(10, 120, size=n)             # e.g., hours used / month
purchase_amount = np.random.randint(20, 800, size=n)           # dollars / month
service_calls = np.random.randint(0, 8, size=n)                # customer support calls

# Create churn probability with a reasonable pattern:
# More service calls + low usage + low purchase -> higher churn probability
base = (
    0.25
    + 0.10 * (service_calls >= 4).astype(int)
    + 0.08 * (monthly_usage < 40).astype(int)
    + 0.08 * (purchase_amount < 150).astype(int)
    - 0.05 * (purchase_amount > 500).astype(int)
)

# Add region effect
region_effect = np.where(regions == "West", 0.03, 0.0) + np.where(regions == "South", 0.02, 0.0)
p_churn = np.clip(base + region_effect + np.random.normal(0, 0.05, size=n), 0.05, 0.90)

churn = (np.random.rand(n) < p_churn).astype(int)

df = pd.DataFrame({
    "age": age,
    "monthly_usage": monthly_usage,
    "purchase_amount": purchase_amount,
    "service_calls": service_calls,
    "region": regions,
    "churn": churn
})

# -----------------------------
# Features / Target
# -----------------------------
X = df.drop(columns=["churn"])
y = df["churn"]

# -----------------------------
# Preprocessing + Model Pipeline
# -----------------------------
num_features = ["age", "monthly_usage", "purchase_amount", "service_calls"]
cat_features = ["region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# -----------------------------
# Predict churn probability for a new customer
# -----------------------------
new_customer = pd.DataFrame([{
    "age": 38,
    "monthly_usage": 35,
    "purchase_amount": 120,
    "service_calls": 5,
    "region": "West"
}])

prob_churn = model.predict_proba(new_customer)[0][1]
pred_label = int(prob_churn >= 0.5)

print("New customer churn probability:", round(prob_churn, 3))
print("At-risk classification (threshold 0.5):", pred_label)

# -----------------------------
# Evaluate on test set
# -----------------------------
y_pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Print coefficients (feature impact)
# -----------------------------
# Get feature names after preprocessing
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cat_features))
feature_names = num_features + cat_names

coef = model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef}).sort_values(by="coefficient", ascending=False)

print("\nTop positive coefficients (increase churn likelihood):")
print(coef_df.head(8).to_string(index=False))

print("\nTop negative coefficients (decrease churn likelihood):")
print(coef_df.tail(8).to_string(index=False))
Add customer churn prediction model
