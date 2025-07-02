import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time

st.set_page_config(page_title="Phoenix Cells - Battery Scanner", page_icon="🔋")

# Load training data
df = pd.read_csv("battery_data.csv")
X = df[['Voltage', 'Temperature', 'Capacity', 'Cycles']]
y = df['Health Status']

model = DecisionTreeClassifier()
model.fit(X, y)

st.title("🔋 Phoenix Cells - Live Battery Health Scanner")
st.markdown("Reviving lithium-ion batteries with AI + Nanotech.")

# Simulate scan data
def simulate_battery_reading():
    voltage = round(np.random.uniform(2.5, 4.2), 2)
    temperature = round(np.random.uniform(20, 60), 1)
    capacity = int(np.random.uniform(30, 100))
    cycles = int(np.random.uniform(100, 1200))
    return voltage, temperature, capacity, cycles

# Live Scan Simulation
if st.button("🔍 Start Battery Scan"):
    with st.spinner("🔄 Scanning battery..."):
        time.sleep(2)
        v, t, c, cyc = simulate_battery_reading()
        st.success("✅ Scan Complete!")
        st.write(f"**Voltage:** {v} V")
        st.write(f"**Temperature:** {t} °C")
        st.write(f"**Capacity:** {c} %")
        st.write(f"**Charge Cycles:** {cyc}")

        prediction = model.predict([[v, t, c, cyc]])
        if prediction[0] == 1:
            st.success("Result: ✅ Battery is Revivable!")
            st.balloons()
        else:
            st.error("Result: ❌ Battery is Dead – Not Revivable.")
