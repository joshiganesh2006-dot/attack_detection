
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Reproducible synthetic training data
rng = np.random.default_rng(42)
rows = []

# Generate normal CAN traffic
for _ in range(1000):
    can_id = rng.choice([0x100, 0x101, 0x102, 0x103, 0x104])
    speed = rng.uniform(0, 120)
    rpm = rng.uniform(800, 4000)
    brake = rng.choice([0, 1], p=[0.8, 0.2])
    throttle = rng.uniform(0, 70)
    temperature = rng.uniform(70, 105)

    # Typical data values for the simulated CAN message IDs
    if can_id == 0x100:
        data = 0
    elif can_id == 0x101:
        data = rpm
    elif can_id == 0x102:
        data = brake
    elif can_id == 0x103:
        data = throttle
    else:
        data = temperature

    rows.append([
        can_id, data, speed, rpm, brake,
        throttle, temperature, 0
    ])

# Generate simulated CAN message-injection examples
for _ in range(1000):
    can_id = 0x100
    data = rng.choice([200, 250, 300, 400, 500])
    speed = rng.uniform(0, 120)
    rpm = rng.uniform(800, 4000)
    brake = rng.choice([0, 1])
    throttle = rng.uniform(0, 70)
    temperature = rng.uniform(70, 105)

    rows.append([
        can_id, data, speed, rpm, brake,
        throttle, temperature, 1
    ])

columns = [
    "can_id", "data", "speed", "rpm", "brake",
    "throttle", "engine_temperature", "label"
]

df = pd.DataFrame(rows, columns=columns)

features = [
    "can_id", "data", "speed", "rpm", "brake",
    "throttle", "engine_temperature"
]

X = df[features]
y = df["label"]

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

joblib.dump(model, "random_forest_model.pkl")

print("Model trained successfully!")
print("Training samples:", len(df))
print("Saved as random_forest_model.pkl")