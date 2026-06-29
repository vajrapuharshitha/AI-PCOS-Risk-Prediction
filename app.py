from textwrap import dedent

app = dedent('''
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import feedparser

st.set_page_config(page_title="AI-Assisted PCOS Risk Prediction", layout="wide")

st.markdown("""
<style>
.stApp{
background: linear-gradient(to bottom,#f7e8ff,#e8f6ff);
}
.block-container{
padding-top:1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open("xgb_pcos_model.pkl","rb"))

model=load_model()

st.title("🧬 AI-Assisted PCOS Risk Prediction System")
st.caption("Developed by Harshitha Vajram")

page=st.sidebar.radio(
"Navigation",
["🏠 Home","📊 Predict","💚 Lifestyle","📰 PCOS Research News","👩‍💻 About"]
)

if page=="🏠 Home":
    st.header("Welcome")
    st.info("AI-assisted screening support for PCOS risk assessment.")
    st.write("Includes biomarkers, lifestyle tips, charts and research updates.")

if page=="📊 Predict":

    st.subheader("Patient Parameters")

    features=[
'Sl_No','Patient_File_No','Age_yrs','Weight_Kg','HeightCm','BMI',
'Blood_Group','Pulse_ratebpm','RR_breathsmin','Hbgdl',
'CycleRI','Cycle_lengthdays','Marraige_Status_Yrs',
'PregnantYN','No_of_aborptions','I___betaHCGmIUmL',
'II____betaHCGmIUmL','FSHmIUmL','LHmIUmL','FSHLH',
'Hipinch','Waistinch','WaistHip_Ratio','TSH_mIUL',
'AMHngmL','PRLngmL','Vit_D3_ngmL','PRGngmL',
'RBSmgdl','Weight_gainYN','hair_growthYN',
'Skin_darkening_YN','Hair_lossYN','PimplesYN',
'Fast_food_YN','RegExerciseYN',
'BP__Systolic_mmHg','BP__Diastolic_mmHg',
'Follicle_No_L','Follicle_No_R',
'Avg_F_size_L_mm','Avg_F_size_R_mm',
'Endometrium_mm'
]

    vals=[]
    c1,c2=st.columns(2)

    for i,f in enumerate(features):
        with c1 if i%2==0 else c2:
            vals.append(st.number_input(f,value=0.0))

    if st.button("Predict Risk"):
        sample=np.array(vals).reshape(1,-1)
        prob=model.predict_proba(sample)[0][1]

        st.header("🧬 Risk Analysis Report")

        st.metric("Risk Score",f"{prob:.1%}")
        st.progress(float(prob))

        if prob<0.35:
            st.success("✅ Low Risk")
        elif prob<0.65:
            st.warning("⚠ Moderate Risk")
        else:
            st.error("🚨 High Risk")

        chart=pd.DataFrame({
            "Feature":["BMI","AMH","LH","Follicles"],
            "Value":[vals[5],vals[24],vals[18],vals[38]]
        })

        st.subheader("📊 Biomarker Dashboard")
        st.bar_chart(chart.set_index("Feature"))

        st.subheader("💚 Personalized Guidance")

        if vals[5]>25:
            st.warning("BMI elevated → regular exercise and balanced meals may help.")
        if vals[24]>5:
            st.info("AMH elevated → discuss hormonal interpretation with healthcare professionals.")
        if vals[35]==0:
            st.info("Regular physical activity may support metabolic health.")

if page=="💚 Lifestyle":
    st.header("Healthy Lifestyle Hub")
    st.success("🥗 Increase fiber-rich foods")
    st.success("🚶 Exercise regularly")
    st.success("😴 Sleep 7–8 hours")
    st.success("💧 Stay hydrated")
    st.success("🧘 Manage stress")

if page=="📰 PCOS Research News":
    st.header("Latest PCOS News")

    feed=feedparser.parse(
    "https://news.google.com/rss/search?q=PCOS"
    )

    for entry in feed.entries[:8]:
        st.write("🔹",entry.title)
        st.write(entry.link)
        st.divider()

if page=="👩‍💻 About":
    st.header("Portfolio")
    st.write("Developer: Harshitha Vajram")
    st.write("Project: AI-Assisted PCOS Risk Prediction")
    st.write("ML + Healthcare + Streamlit")
''')

req = """streamlit
numpy
pandas
scikit-learn
xgboost
feedparser
"""

open("/mnt/data/app.py","w").write(app)
open("/mnt/data/requirements.txt","w").write(req)

print("files created")
