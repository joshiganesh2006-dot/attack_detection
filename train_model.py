
import random
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

random.seed(42)

FEATURES = [
    "ecu_id",
    "can_id",
    "data",
    "speed",
    "rpm",
    "brake",
    "throttle",
    "engine_temperature",
    "message_rate",
    "repeated_message",
    "sensor_inconsistent",
]

ATTACKS = {
    0: "Normal Traffic",
    1: "CAN Message Injection",
    2: "Denial-of-Service (DoS)",
    3: "Replay Attack",
    4: "Sensor Spoofing",
}

rows = []

# Create synthetic examples for each class.
# These are simulation patterns, not real vehicle data.

for _ in range(1200):
    ecu_id = random.choice([1, 2, 3, 4])
    can_id = random.choice([256, 257, 258, 259, 260])
    speed = random.randint(0, 120)
    rpm = random.randint(800, 4000)
    brake = random.choice([0, 1])
    throttle = random.randint(0, 70)
    temperature = random.randint(70, 105)
    data = random.randint(0, 100)
    rate = random.randint(5, 30)
    repeated = 0
    inconsistent = 0

    rows.append([
        ecu_id, can_id, data, speed, rpm, brake,
        throttle, temperature, rate, repeated, inconsistent, 0
    ])

for _ in range(800):
    # CAN message injection: unusual data pattern
    rows.append([
        random.choice([1, 2, 3, 4]),
        random.choice([256, 257, 258, 259, 260]),
        random.randint(200, 1000),
        random.randint(0, 120),
        random.randint(800, 4000),
        random.choice([0, 1]),
        random.randint(0, 70),
        random.randint(70, 105),
        random.randint(5, 30), 0, 0, 1
    ])

    # DoS: unusually high message rate
    rows.append([
        random.choice([1, 2, 3, 4]),
        random.choice([256, 257, 258, 259, 260]),
        random.randint(0, 100),
        random.randint(0, 120),
        random.randint(800, 4000),
        random.choice([0, 1]),
        random.randint(0, 70),
        random.randint(70, 105),
        random.randint(100, 300), 0, 0, 2
    ])

    # Replay: repeated messages
    rows.append([
        random.choice([1, 2, 3, 4]),
        random.choice([256, 257, 258, 259, 260]),
        random.randint(0, 100),
        random.randint(0, 120),
        random.randint(800, 4000),
        random.choice([0, 1]),
        random.randint(0, 70),
        random.randint(70, 105),
        random.randint(5, 30), 1, 0, 3
    ])

    # Sensor spoofing: inconsistent sensor readings
    rows.append([
        random.choice([1, 2, 3, 4]),
        random.choice([256, 257, 258, 259, 260]),
        random.randint(0, 100),
        random.choice([random.randint(0, 20), random.randint(180, 300)]),
        random.choice([random.randint(800, 1200), random.randint(6000, 9000)]),
        random.choice([0, 1]),
        random.randint(0, 100),
        random.choice([random.randint(0, 40), random.randint(130, 180)]),
        random.randint(5, 30), 0, 1, 4
    ])

columns = FEATURES + ["label"]
df = pd.DataFrame(rows, columns=columns)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model training completed.")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions, labels=list(ATTACKS.keys()),
    target_names=list(ATTACKS.values()), zero_division=0
))

joblib.dump(model, "random_forest_model.pkl")
print("Saved model as random_forest_model.pkl")
