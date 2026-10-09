import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Attack Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle CAN Attack Detection System")
st.write(
    "Enter vehicle and CAN message details to check "
    "whether the Random Forest model identifies an attack."
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Enter CAN and Vehicle Information")

with st.form("vehicle_input_form"):

    col1, col2 = st.columns(2)

    with col1:
        can_id = st.text_input(
            "CAN ID (example: 0x100)",
            value="0x100"
        )

        can_data = st.number_input(
            "CAN Data",
            min_value=0,
            max_value=4294967295,
            value=100
        )

        speed = st.number_input(
            "Speed (km/h)",
            min_value=0,
            max_value=300,
            value=45
        )

        rpm = st.number_input(
            "Engine RPM",
            min_value=0,
            max_value=15000,
            value=1800
        )

    with col2:
        brake = st.number_input(
            "Brake",
            min_value=0,
            max_value=100,
            value=0
        )

        throttle = st.number_input(
            "Throttle",
            min_value=0,
            max_value=100,
            value=20
        )

        engine_temperature = st.number_input(
            "Engine Temperature (°C)",
            min_value=-40.0,
            max_value=200.0,
            value=80.0
        )

    submitted = st.form_submit_button(
        "🔍 Detect Attack",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    try:
        # Convert hexadecimal CAN ID into a number.
        # This matches the preprocessing used in training.
        numeric_can_id = int(can_id.strip(), 16)

        if numeric_can_id < 0 or numeric_can_id > 2047:
            st.error("Enter a valid standard 11-bit CAN ID.")
            st.stop()

        # Use the same feature names and order as training.
        features = pd.DataFrame([{
            "can_id": numeric_can_id,
            "data": can_data,
            "speed": speed,
            "rpm": rpm,
            "brake": brake,
            "throttle": throttle,
            "engine_temperature": engine_temperature
        }])

        prediction = int(model.predict(features)[0])

        st.divider()
        st.subheader("Detection Result")

        if prediction == 1:

            st.error("🚨 ATTACK DETECTED")

            st.write("**Attack Type:** CAN Message Injection")
            st.write(f"**CAN ID:** {can_id}")
            st.write(f"**CAN Data:** {can_data}")

            st.warning(
                "The trained model classified this input as an attack."
            )

        else:

            st.success("🟢 NORMAL TRAFFIC")

            st.write("**Attack Type:** None detected")
            st.write(f"**CAN ID:** {can_id}")
            st.write(f"**CAN Data:** {can_data}")

            st.info(
                "The trained model classified this input as normal."
            )

    except ValueError:
        st.error(
            "Invalid CAN ID. Enter a hexadecimal value such as 0x100."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
