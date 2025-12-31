import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn



# =========================
# Step 5: Load & Explore Data
# =========================

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Preview first 5 rows
print(df.head())

# Check class distribution
print(df["Class"].value_counts())



# =========================
# Step 6: Split Features & Labels
# =========================

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split (keep class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Step 7: Handle Class Imbalance (SMOTE)
# =========================

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)

# Check class balance before and after SMOTE
print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE :", np.bincount(y_train_resampled))





# Step 8: Train Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)





# Step 9: Model Evaluation

y_scores = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

avg_precision = average_precision_score(y_test, y_scores)
print("Average Precision Score:", avg_precision)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)





# Step 10: MLflow Experiment Tracking

mlflow.set_experiment("Credit Card Fraud Detection")

with mlflow.start_run():

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("average_precision", avg_precision)

    mlflow.sklearn.log_model(model, "fraud_model")

    print("MLflow run logged successfully!")

