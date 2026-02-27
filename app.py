import streamlit as st
import numpy as np
import pickle

# Load model & preprocessing objects
model = pickle.load(open("energy_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_tool = pickle.load(open("le_tool.pkl", "rb"))
le_process = pickle.load(open("le_process.pkl", "rb"))
le_shift = pickle.load(open("le_shift.pkl", "rb"))

# Page config
st.set_page_config(page_title="Green Energy Predictor", layout="wide")

# Custom CSS Styling
# Page config
st.set_page_config(page_title="Green Energy Predictor", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        .main-title {
            font-size: 52px;
            font-weight: 900;
            text-align: center;
            background: linear-gradient(90deg, #00C9A7, #00BFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }
        .sub-title {
            font-size: 30px;   /* Increased size */
            font-weight: 700;  /* Bold */
            text-align: center;
            color: #1F4E79;    /* Strong visible blue */
            margin-bottom: 50px;
        }
        .stButton>button {
            background-color: #00C9A7;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #00BFFF;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">⚡ Fabrication Energy Optimization Model</div>', unsafe_allow_html=True)

# Larger, More Visible Subtitle
st.markdown('<div class="sub-title">Predict High Energy Usage for Sustainable Semiconductor Manufacturing</div>', unsafe_allow_html=True)


#st.markdown("### Predict High Energy Usage for Sustainable Semiconductor Manufacturing")

st.sidebar.header("Input Parameters")

tool_id = st.sidebar.selectbox("Tool ID", le_tool.classes_)
process_type = st.sidebar.selectbox("Process Type", le_process.classes_)
shift = st.sidebar.selectbox("Shift", le_shift.classes_)

wafers_processed = st.sidebar.number_input("Wafers Processed", 0, 500)
runtime_hours = st.sidebar.number_input("Runtime Hours", 0.0, 24.0)
idle_hours = st.sidebar.number_input("Idle Hours", 0.0, 24.0)
cleanroom_temp = st.sidebar.number_input("Cleanroom Temp (°C)", 15.0, 30.0)
humidity = st.sidebar.number_input("Humidity (%)", 20.0, 70.0)

if st.sidebar.button("Predict"):

    tool_id_enc = le_tool.transform([tool_id])[0]
    process_enc = le_process.transform([process_type])[0]
    shift_enc = le_shift.transform([shift])[0]

    features = np.array([[tool_id_enc, process_enc, wafers_processed,
                          runtime_hours, idle_hours, cleanroom_temp,
                          humidity, shift_enc]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Energy Consumption Predicted")
        st.markdown("### 💡 Recommendation:")
        st.write("- Optimize runtime")
        st.write("- Reduce idle hours")
        st.write("- Monitor cleanroom temperature")
    else:
        st.success("✅ Energy Consumption Within Sustainable Limit")