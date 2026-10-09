
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ECU Security Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Intelligent ECU Authentication and Cyber Threat Detection")
st.markdown(
    "### In-Vehicle Network Security Monitoring System"
)
st.write(
    "This software-only simulation demonstrates ECU authentication "
    "and Random Forest-based detection of simulated vehicle network attacks."
)

# --------------------------------------------------
# LOAD TRAINED RANDOM FOREST MODEL
# --------------------------------------------------

MODEL_PATH = "random_forest_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("Model file not found: random_forest_model.pkl")
    st.info(
        "Open your project terminal and run: "
        "python train_model.py"
    )
    st.stop()

# --------------------------------------------------
# ECU AUTHENTICATION DETAILS
# --------------------------------------------------

ecu_registry = {
    "Engine ECU": {
        "ecu_id": 1,
        "password": "ENG123"
    },
    "Brake ECU": {
        "ecu_id": 2,
        "password": "BRK123"
    },
    "Transmission ECU": {
        "ecu_id": 3,
        "password": "TRN123"
    },
    "Body Control ECU": {
        "ecu_id": 4,
        "password": "BCM123"
    }
}

# --------------------------------------------------
# SIDEBAR: ECU AUTHENTICATION
# --------------------------------------------------

st.sidebar.header("🔐 ECU Authentication")

ecu_name = st.sidebar.selectbox(
    "Select ECU",
    [
        "Engine ECU",
        "Brake ECU",
        "Transmission ECU",
        "Body Control ECU",
        "Unknown ECU"
    ]
)

password = st.sidebar.text_input(
    "Enter ECU Authentication Key",
    type="password"
)

authenticate_clicked = st.sidebar.button("Authenticate ECU")

if authenticate_clicked:
    if (
        ecu_name in ecu_registry
        and password == ecu_registry[ecu_name]["password"]
    ):
        st.session_state["authenticated"] = True
        st.session_state["authenticated_ecu"] = ecu_name
        st.sidebar.success("ECU Authentication Successful")
    else:
        st.session_state["authenticated"] = False
        st.session_state["authenticated_ecu"] = None
        st.sidebar.error("Authentication Failed: Unauthorized ECU")

if st.session_state.get("authenticated", False):
    st.sidebar.success(
        "Authenticated: "
        + st.session_state.get("authenticated_ecu", "")
    )
else:
    st.sidebar.warning("ECU not authenticated")

# --------------------------------------------------
# ATTACK SCENARIO SELECTION
# --------------------------------------------------

st.subheader("🛡️ Vehicle Network Simulation")

scenario = st.selectbox(
    "Select a scenario to simulate",
    [
        "Normal Traffic",
        "CAN Message Injection",
        "Denial-of-Service (DoS)",
        "Replay Attack",
        "Sensor Spoofing"
    ]
)

st.caption(
    "Choose a scenario and click Detect Threat to test the "
    "model using simulated network data."
)

# --------------------------------------------------
# DEFAULT VEHICLE AND NETWORK VALUES
# --------------------------------------------------

ecu_id = 1
can_id = 256
data = 100
speed = 60
rpm = 2000
brake = 0
throttle = 30
engine_temperature = 90
message_rate = 20
repeated_message = 0
sensor_inconsistent = 0

# --------------------------------------------------
# SIMULATE EACH SCENARIO
# --------------------------------------------------

if scenario == "Normal Traffic":
    ecu_id = 1
    can_id = 256
    data = 100
    speed = 60
    rpm = 2000
    brake = 0
    throttle = 30
    engine_temperature = 90
    message_rate = 20
    repeated_message = 0
    sensor_inconsistent = 0

elif scenario == "CAN Message Injection":
    ecu_id = 5
    can_id = 999
    data = 400
    speed = 60
    rpm = 2000
    brake = 0
    throttle = 30
    engine_temperature = 90
    message_rate = 40
    repeated_message = 0
    sensor_inconsistent = 0

elif scenario == "Denial-of-Service (DoS)":
    ecu_id = 5
    can_id = 256
    data = 100
    speed = 60
    rpm = 2000
    brake = 0
    throttle = 30
    engine_temperature = 90
    message_rate = 200
    repeated_message = 0
    sensor_inconsistent = 0

elif scenario == "Replay Attack":
    ecu_id = 5
    can_id = 256
    data = 100
    speed = 60
    rpm = 2000
    brake = 0
    throttle = 30
    engine_temperature = 90
    message_rate = 20
    repeated_message = 1
    sensor_inconsistent = 0

elif scenario == "Sensor Spoofing":
    ecu_id = 5
    can_id = 256
    data = 100
    speed = 250
    rpm = 8000
    brake = 0
    throttle = 30
    engine_temperature = 160
    message_rate = 20
    repeated_message = 0
    sensor_inconsistent = 1

# --------------------------------------------------
# DISPLAY SIMULATED INPUT DATA
# --------------------------------------------------

st.subheader("📊 Simulated Vehicle Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Vehicle Speed", f"{speed} km/h")
    st.metric("Engine RPM", rpm)
    st.metric("CAN ID", can_id)

with col2:
    st.metric("Throttle", f"{throttle}%")
    st.metric("Brake", brake)
    st.metric("Engine Temperature", f"{engine_temperature} °C")

with col3:
    st.metric("Message Rate", f"{message_rate} messages/sec")
    st.metric("Repeated Message", repeated_message)
    st.metric("Sensor Inconsistency", sensor_inconsistent)

# --------------------------------------------------
# DETECTION BUTTON
# --------------------------------------------------

if st.button("🔍 Detect Threat", type="primary"):

    # Determine whether the selected ECU is authorized
    is_authenticated = (
        st.session_state.get("authenticated", False)
        and st.session_state.get("authenticated_ecu") == ecu_name
        and ecu_name in ecu_registry
    )

    st.subheader("🔐 ECU Authentication Result")

    if is_authenticated:
        st.success(f"{ecu_name} is authenticated.")
    else:
        st.error(
            "ECU is NOT authenticated. Treat this ECU as unauthorized."
        )

    # Use the registered ECU ID for authorized ECUs.
    # Unknown or unauthenticated ECUs are represented by ID 5.
    if is_authenticated:
        ecu_id = ecu_registry[ecu_name]["ecu_id"]
    else:
        ecu_id = 5

    # IMPORTANT: These 11 feature names and their order must match
    # the features used to train random_forest_model.pkl.
    feature_columns = [
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
        "sensor_inconsistent"
    ]

    sample = pd.DataFrame(
        [[
            ecu_id,
            can_id,
            data,
            speed,
            rpm,
            brake,
            throttle,
            engine_temperature,
            message_rate,
            repeated_message,
            sensor_inconsistent
        ]],
        columns=feature_columns
    )

    # --------------------------------------------------
    # RANDOM FOREST PREDICTION
    # --------------------------------------------------

    prediction = int(model.predict(sample)[0])

    attack_names = {
        0: "Normal Traffic",
        1: "CAN Message Injection",
        2: "Denial-of-Service (DoS)",
        3: "Replay Attack",
        4: "Sensor Spoofing"
    }

    detected_attack = attack_names.get(
        prediction,
        f"Unknown Class ({prediction})"
    )

    st.subheader("🚨 Threat Detection Result")

    if not is_authenticated:
        st.error("SECURITY ALERT: Unauthorized ECU detected!")

    if prediction == 0:
        if is_authenticated:
            st.success("No attack detected by the model.")
        else:
            st.warning(
                "The model predicted normal traffic, but ECU "
                "authentication failed. The ECU remains unauthorized."
            )
    else:
        st.error(f"Potential Attack Detected: {detected_attack}")

    # --------------------------------------------------
    # SHOW MODEL INPUT
    # --------------------------------------------------

    st.subheader("🧾 Data Sent to the Random Forest Model")
    st.dataframe(sample, use_container_width=True)

    st.caption(
        "This is a synthetic demonstration. Predictions depend on "
        "the training dataset and are not proof of real-world attacks."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Academic Project | ECU Authentication | Random Forest "
    "Cyber Threat Detection | Software Simulation"
)
