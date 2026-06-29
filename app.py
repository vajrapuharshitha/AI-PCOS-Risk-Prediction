import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="AI-Assisted PCOS Risk Prediction")

model = pickle.load(open("xgb_pcos_model.pkl","rb"))

st.title("🧬 AI-Assisted PCOS Risk Prediction")
st.subheader("Developed by Harshitha Vajram")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose",
    ["Home","Predict","About"]
)

if page=="Home":
    st.write("Welcome to AI-assisted PCOS risk prediction system")
    st.info("Educational screening support only")

feature_names=[
'Sl_No','Patient_File_No','Age_yrs','Weight_Kg',
'HeightCm','BMI','Blood_Group','Pulse_ratebpm',
'RR_breathsmin','Hbgdl','CycleRI',
'Cycle_lengthdays','Marraige_Status_Yrs',
'PregnantYN','No_of_aborptions',
'I___betaHCGmIUmL','II____betaHCGmIUmL',
'FSHmIUmL','LHmIUmL','FSHLH',
'Hipinch','Waistinch','WaistHip_Ratio',
'TSH_mIUL','AMHngmL','PRLngmL',
'Vit_D3_ngmL','PRGngmL','RBSmgdl',
'Weight_gainYN','hair_growthYN',
'Skin_darkening_YN','Hair_lossYN',
'PimplesYN','Fast_food_YN',
'RegExerciseYN','BP__Systolic_mmHg',
'BP__Diastolic_mmHg','Follicle_No_L',
'Follicle_No_R','Avg_F_size_L_mm',
'Avg_F_size_R_mm','Endometrium_mm'
]

if page=="Predict":

    inputs=[]

    for f in feature_names:
        x=st.number_input(f,value=0.0)
        inputs.append(x)

    if st.button("Predict Risk"):

        sample=np.array(inputs).reshape(1,-1)

        prob=model.predict_proba(sample)[0][1]

        st.write("Risk Score:",round(prob,3))

        if prob<0.35:
            st.success("✅ Low Risk")

        elif prob<0.65:
            st.warning("⚠ Moderate Risk")

        else:
            st.error("🚨 High Risk")

if page=="About":
    st.write("Project: AI-Assisted PCOS Risk Prediction")
    st.write("Model: XGBoost")
    st.write("Developer: Harshitha Vajram")
