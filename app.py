import streamlit as st
import numpy as np
import pandas as pd
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
background: linear-gradient(to bottom,#f8e8ff,#e7f5ff);
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 2px 10px rgba(0,0,0,0.1);
margin-bottom:20px;
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
[
"🏠 Home",
"📊 Predict",
"💚 Lifestyle",
"🧪 Biomarkers",
"👩‍💻 About"
]
)


# HOME

if page=="🏠 Home":

    st.markdown("""
    <div class='card'>
    <h2>Welcome</h2>

    AI-assisted screening support for PCOS risk assessment.
    Includes biomarkers, visual reports and guidance.

    </div>
    """,unsafe_allow_html=True)

    st.subheader("📰 Research Updates")

    st.info(
    "Research continues exploring hormonal patterns, genetics and metabolic biomarkers in PCOS."
    )

    st.info(
    "Lifestyle interventions remain an active area of study."
    )


# PREDICT

if page=="📊 Predict":

    feature_names=[

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

    values=[]

    c1,c2=st.columns(2)

    for i,f in enumerate(feature_names):

        with c1 if i%2==0 else c2:

            x=st.number_input(
                f,
                value=0.0
            )

            values.append(x)


    if st.button("Predict Risk"):

        sample=np.array(values).reshape(1,-1)

        prob=model.predict_proba(sample)[0][1]

        st.header("🧬 PCOS Risk Report")

        st.metric(
            "Risk Score",
            f"{prob:.1%}"
        )

        st.progress(float(prob))

        if prob<0.35:

            st.success(
                "✅ Low Risk"
            )

        elif prob<0.65:

            st.warning(
                "⚠ Moderate Risk"
            )

        else:

            st.error(
                "🚨 High Risk"
            )

        st.subheader(
            "📊 Biomarker Dashboard"
        )

        chart=pd.DataFrame({

        "Feature":[
        "BMI",
        "AMH",
        "LH",
        "Follicles"
        ],

        "Value":[
        values[5],
        values[24],
        values[18],
        values[38]
        ]

        })

        st.bar_chart(
        chart.set_index("Feature")
        )


        st.subheader(
        "💚 Guidance"
        )

        if values[5]>25:

            st.warning(
            "BMI elevated: balanced nutrition and regular exercise may help."
            )

        if values[24]>5:

            st.info(
            "AMH elevated compared with common ranges."
            )


# LIFESTYLE

if page=="💚 Lifestyle":

    st.header(
    "Healthy Lifestyle Hub"
    )

    st.success(
    "🥗 Balanced meals"
    )

    st.success(
    "🚶 Exercise regularly"
    )

    st.success(
    "😴 Sleep 7–8 hours"
    )

    st.success(
    "💧 Stay hydrated"
    )

    st.success(
    "🧘 Manage stress"
    )


# BIOMARKERS

if page=="🧪 Biomarkers":

    st.header(
    "Biomarker Information"
    )

    with st.expander("AMH"):
        st.write(
        "Anti-Müllerian Hormone"
        )

    with st.expander("LH"):
        st.write(
        "Luteinizing Hormone"
        )

    with st.expander("BMI"):
        st.write(
        "Body Mass Index"
        )


# ABOUT

if page=="👩‍💻 About":

    st.header(
    "Developer Portfolio"
    )

    st.write(
    "Harshitha Vajram"
    )

    st.write(
    "AI-Assisted PCOS Risk Prediction Project"
    )
