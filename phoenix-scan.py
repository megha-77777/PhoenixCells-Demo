import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
        st.write(f"**Voltage:** {v} V")
        st.write(f"**Temperature:** {t} °C")
        st.write(f"**Capacity:** {c} %")
        st.write(f"**Charge Cycles:** {cyc}")
        st.write(f"**AI Confidence Score:** {round(proba * 100, 2)}%")

        # Save to session log
        st.session_state.scan_log.append({
            "Voltage (V)": v,
            "Temperature (°C)": t,
            "Capacity (%)": c,
            "Cycles": cyc,
            "Prediction": result,
            "Confidence (%)": round(proba * 100, 2)
        })

# Display scan history
if st.session_state.scan_log:
    st.markdown("---")
    st.subheader("📄 Scan History")
    history_df = pd.DataFrame(st.session_state.scan_log)
    st.dataframe(history_df, use_container_width=True)

    # Export report
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Report (CSV)", csv, "battery_scan_report.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("Phoenix Cells Demo | Team Catalyst Nexus")
