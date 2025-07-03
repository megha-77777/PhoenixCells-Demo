import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
from PIL import Image
import pytesseract
import time
import io

# App setup
st.set_page_config(page_title="Phoenix Cells - Smart Battery Scanner", page_icon="🔋")

# Load training data
df = pd.read_csv("battery_data.csv")
X = df[['Voltage', 'Temperature', 'Capacity', 'Cycles']]
y = df['Health Status']
model = DecisionTreeClassifier()
model.fit(X, y)

# Session state for scan history
if "scan_log" not in st.session_state:
    st.session_state.scan_log = []

# Title
st.title("🔋 Phoenix Cells - Smart Battery Revival Scanner")
st.caption("Powered by AI + Nanotech")

# Simulate sensor reading
def simulate_battery_reading():
    voltage = round(np.random.uniform(2.5, 4.2), 2)
    temperature = round(np.random.uniform(20, 60), 1)
    capacity = int(np.random.uniform(30, 100))
    cycles = int(np.random.uniform(100, 1200))
    return voltage, temperature, capacity, cycles

# Scan process
if st.button("🔍 Start Battery Scan"):
    with st.spinner("⏳ Scanning battery..."):
        time.sleep(2)
        v, t, c, cyc = simulate_battery_reading()
        prediction = model.predict([[v, t, c, cyc]])[0]
        proba = model.predict_proba([[v, t, c, cyc]])[0][prediction]

        result = "Revivable" if prediction == 1 else "Dead"
        emoji = "✅" if prediction == 1 else "❌"

        # Show results
        st.success(f"{emoji} Scan Complete - Battery is **{result}**")
        st.markdown("### 🔧 Battery Vitals")
        col1, col2 = st.columns(2)
        col1.metric("🔋 Voltage", f"{v} V")
        col2.metric("🌡 Temperature", f"{t} °C")
        st.progress(min(c, 100))  # capacity bar

        st.markdown(f"**📈 Charge Cycles:** `{cyc}`")
        st.markdown(f"**🤖 AI Confidence:** `{round(proba * 100, 2)} %`")

        # Save to scan history
        st.session_state.scan_log.append({
            "Voltage (V)": v,
            "Temperature (°C)": t,
            "Capacity (%)": c,
            "Cycles": cyc,
            "Prediction": result,
            "Confidence (%)": round(proba * 100, 2)
        })

# OCR Camera Scanner
st.markdown("---")
st.subheader("📷 Scan Battery Label (OCR)")
uploaded_image = st.camera_input("Take a photo of battery label")

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    text = pytesseract.image_to_string(img)
    st.markdown("**🔍 Scanned Text from Battery Label:**")
    st.code(text)

# Scan History Table + Downloads
if st.session_state.scan_log:
    st.markdown("---")
    st.subheader("📄 Scan History")
    history_df = pd.DataFrame(st.session_state.scan_log)
    st.dataframe(history_df, use_container_width=True)

    # CSV Download
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Report (CSV)", csv, "battery_scan_report.csv", "text/csv")

    # PDF Download
    class PDF(FPDF)
