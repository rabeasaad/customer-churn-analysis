import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Customer Churn Dataset
np.random.seed(42)
n_samples = 1000

data = {
    "CustomerID": np.arange(1001, 1001 + n_samples),
    "Tenure_Months": np.random.randint(1, 61, size=n_samples),
    "Monthly_Charges": np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2),
    "Contract_Type": np.random.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        size=n_samples,
        p=[0.5, 0.3, 0.2],
    ),
    "Payment_Method": np.random.choice(
        ["Electronic Check", "Credit Card", "Bank Transfer"], size=n_samples
    ),
    "Paperless_Billing": np.random.choice(["Yes", "No"], size=n_samples),
}

df = pd.DataFrame(data)

# Define Churn Probability Logic
churn_prob = (
    0.4 * (df["Contract_Type"] == "Month-to-Month")
    + 0.3 * (df["Payment_Method"] == "Electronic Check")
    - 0.005 * df["Tenure_Months"]
    + 0.002 * df["Monthly_Charges"]
)
churn_prob = np.clip(churn_prob, 0.05, 0.95)
df["Churn"] = np.random.binomial(1, churn_prob)

print("--- Sample Data Generated ---")
print(df.head())

# 2. Data Preprocessing
df_encoded = pd.get_dummies(
    df.drop("CustomerID", axis=1),
    columns=["Contract_Type", "Payment_Method", "Paperless_Billing"],
    drop_first=True,
)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))