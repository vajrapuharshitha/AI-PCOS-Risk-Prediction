import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("xgb_pcos_model.pkl","rb"))

st.title("🧬 PCOS Risk Prediction System")
st.write("Enter patient details")

feature_names = [
'Sl_No',
'Patient_File_No',
'Age_yrs',
'Weight_Kg',
'HeightCm',
'BMI',
'Blood_Group',
'Pulse_ratebpm',
'RR_breathsmin',
'Hbgdl',
'CycleRI',
'Cycle_lengthdays',
'Marraige_Status_Yrs',
'PregnantYN',
'No_of_aborptions',
'I___betaHCGmIUmL',
'II____betaHCGmIUmL',
'FSHmIUmL',
'LHmIUmL',
'FSHLH',
'Hipinch',
'Waistinch',
'WaistHip_Ratio',
'TSH_mIUL',
'AMHngmL',
'PRLngmL',
'Vit_D3_ngmL',
'PRGngmL',
'RBSmgdl',
'Weight_gainYN',
'hair_growthYN',
'Skin_darkening_YN',
'Hair_lossYN',
'PimplesYN',
'Fast_food_YN',
'RegExerciseYN',
'BP__Systolic_mmHg',
'BP__Diastolic_mmHg',
'Follicle_No_L',
'Follicle_No_R',
'Avg_F_size_L_mm',
'Avg_F_size_R_mm',
'Endometrium_mm'
]

inputs=[]

for feature in feature_names:
    val=st.number_input(feature,value=0.0)
    inputs.append(val)

if st.button("Predict"):

    sample=np.array(inputs).reshape(1,-1)

    prob=model.predict_proba(sample)[0][1]

    st.subheader(f"Risk Score: {prob:.3f}")

    if prob>=0.35:
        st.error("⚠ High PCOS Risk")
    else:
        st.success("✅ Low PCOS Risk")
