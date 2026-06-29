import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="AI-Assisted PCOS Risk Prediction",
    page_icon="🧬",
    layout="wide"
)

# Theme
st.markdown("""
<style>
.stApp{
background: linear-gradient(to bottom,#f7e8ff,#e8f6ff);
}

.box{
padding:20px;
border-radius:15px;
background:white;
margin:10px;
box-shadow:2px 2px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

model=pickle.load(open("xgb_pcos_model.pkl","rb"))

st.title("🧬 AI-Assisted PCOS Risk Prediction System")
st.caption("Developed by Harshitha Vajram")

page=st.sidebar.selectbox(
"Navigation",
["🏠 Home","📊 Predict","💚 Lifestyle","🧪 Biomarkers","👩‍💻 About"]
)

if page=="🏠 Home":

    st.markdown(
    """
    <div class='box'>
    <h2>Welcome</h2>
    Early AI-assisted screening support for PCOS risk assessment.
    </div>
    """,
    unsafe_allow_html=True
    )

    st.subheader("📰 PCOS Research Updates")

    st.info("Researchers continue studying genetics, metabolism and biomarker patterns in PCOS.")
    st.info("Lifestyle interventions and hormonal biomarkers remain active areas of research.")
    st.info("Emerging studies explore gut microbiome and metabolic pathways.")

if page=="💚 Lifestyle":

    st.header("Healthy Lifestyle Hub")

    st.success("🥗 Eat fiber-rich foods and balanced meals")
    st.success("🚶 Regular exercise can improve metabolic health")
    st.success("😴 Aim for consistent sleep")
    st.success("💧 Stay hydrated")
    st.success("🧘 Stress management matters")

if page=="🧪 Biomarkers":

    st.header("Biomarker Education")

    with st.expander("AMH"):
        st.write("Anti-Müllerian Hormone: may be elevated in some PCOS patterns.")

    with st.expander("LH/FSH"):
        st.write("Hormones related to ovulation patterns.")

    with st.expander("BMI"):
        st.write("Body mass index can influence metabolic risk.")

if page=="📊 Predict":

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

    inputs=[]

    for f in feature_names:
        x=st.number_input(f,value=0.0)
        inputs.append(x)

    if st.button("Predict"):

        sample=np.array(inputs).reshape(1,-1)

        prob=model.predict_proba(sample)[0][1]

        st.metric("Risk Score",f"{prob:.2%}")

        if prob<0.35:
            st.success("✅ Low Risk")

        elif prob<0.65:
            st.warning("⚠ Moderate Risk")

        else:
            st.error("🚨 High Risk")

if page=="👩‍💻 About":

    st.header("Developer Portfolio")

    st.write("Harshitha Vajram")
    st.write("AI-Assisted PCOS Risk Prediction Project")
    st.write("Machine Learning + Healthcare + Streamlit")
