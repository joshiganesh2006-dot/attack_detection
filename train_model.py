import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("combined_dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))


# ============================================================
# CONVERT CAN ID TO NUMBER
# ============================================================

data["can_id"] = data["can_id"].apply(
    lambda x: int(str(x), 16)
)


# ============================================================
# FEATURES USED BY RANDOM FOREST
# ============================================================

features = [
    "can_id",
    "data",
    "speed",
    "rpm",
    "brake",
    "throttle",
    "engine_temperature"
]

X = data[features]
y = data["label"]


# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)


# ============================================================
# TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("       RANDOM FOREST RESULTS")
print("==============================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"]
    )
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

joblib.dump(model, "random_forest_model.pkl")

print("==============================================")
print("Model saved as:")
print("random_forest_model.pkl")
print("==============================================")