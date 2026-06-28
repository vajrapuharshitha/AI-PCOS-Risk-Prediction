import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("xgb_pcos_model.pkl", "rb"))

st.title("🧬 PCOS Risk Prediction System")

st.write("Enter patient details below:")

features = []

# IMPORTANT: You may need to adjust number of features to match your model
for i in range(6):
    val = st.number_input(f"Feature {i+1}")
    features.append(val)

if st.button("Predict"):
    sample = np.array(features).reshape(1, -1)
    prob = model.predict_proba(sample)[0][1]

    st.write("### Risk Score:", prob)

    if prob > 0.35:
        st.error("⚠️ High Risk of PCOS")
    else:
        st.success("✅ Low Risk")
