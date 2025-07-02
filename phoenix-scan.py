import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Phoenix Cells Scanner", page_icon="🔋")

# Load mock dataset
df = pd.read_csv("battery_data.csv")
X = df[['Voltage', 'Temperature', 'Capacity', 'Cycles']]
y = df['Health Status']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# UI
st.title("🔋 Phoenix Cells - AI Battery Health Scanner")
st.markdown("Scan your lithium-ion battery and check if it's revivable!")

voltage = st.slider("Voltage (V)", 2.5, 4.2, 3.6)
temperature = st.slider("Temperature (°C)", 10, 60, 30)
capacity = st.slider("Capacity (%)", 0, 100, 70)
cycles = st.slider("Charge Cycles", 0, 1500, 400)

if st.button("🔍 Scan Battery"):
    prediction = model.predict([[voltage, temperature, capacity, cycles]])
    if prediction[0] == 1:
        st.success("✅ Battery is Revivable!")
        st.balloons()
    else:
        st.error("❌ Battery is Dead – Not Revivable.")
