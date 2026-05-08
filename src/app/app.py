"""
STREAMLIT WEB INTERFACE - Frontend UI
=====================================
Run this with: streamlit run app.py
"""

import streamlit as st
import requests

# === UI CONFIGURATION ===
st.set_page_config(
    page_title="Medicare Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# FastAPI Backend URL
API_URL = "http://localhost:8000/predict"

# === HEADER ===
st.title("🛡️ Medicare Fraud Detection")
st.markdown("""
Evaluate inpatient Medicare claims for structural and attributive anomalies using our **Isolation Forest** model. 
Fill out the claim details below to run the analysis.
""")
st.divider()

# === INPUT FORM ===
# Using a form prevents the app from rerunning every time a user types a single letter
with st.form("claim_form"):
    
    st.subheader("1. Provider & Physician Details")
    col1, col2 = st.columns(2)
    with col1:
        prvdr_num = st.text_input("Provider Number (PRVDR_NUM)", value="1002CR")
        at_physn_npi = st.text_input("Attending Physician NPI", value="9876543210")
    with col2:
        op_physn_npi = st.text_input("Operating Physician NPI", value="missing")
        ot_physn_npi = st.text_input("Other Physician NPI", value="missing")

    st.subheader("2. Financials & Utilization")
    col3, col4 = st.columns(2)
    with col3:
        nch_prmry_pyr_clm_pd_amt = st.number_input("Primary Payer Claim Paid Amount ($)", min_value=0.0, value=0.0, step=100.0)
    with col4:
        clm_utlztn_day_cnt = st.number_input("Utilization Day Count", min_value=0, value=1, step=1)

    st.subheader("3. Diagnosis & Procedure Codes")
    col5, col6, col7 = st.columns(3)
    with col5:
        admtng_icd9_dgns_cd = st.text_input("Admitting ICD-9 Code", value="41401")
    with col6:
        clm_drg_cd = st.text_input("Claim DRG Code", value="101")
    with col7:
        icd9_prcdr_cd_1 = st.text_input("Procedure Code 1", value="3615")

    st.divider()
    
    # Submit Button
    submitted = st.form_submit_button("Run Fraud Detection", type="primary", use_container_width=True)

# === PREDICTION LOGIC ===
if submitted:
    # Construct the payload matching the FastAPI schema
    payload = {
        "PRVDR_NUM": prvdr_num,
        "NCH_PRMRY_PYR_CLM_PD_AMT": nch_prmry_pyr_clm_pd_amt,
        "AT_PHYSN_NPI": at_physn_npi,
        "OP_PHYSN_NPI": op_physn_npi,
        "OT_PHYSN_NPI": ot_physn_npi,
        "CLM_UTLZTN_DAY_CNT": clm_utlztn_day_cnt,
        "ADMTNG_ICD9_DGNS_CD": admtng_icd9_dgns_cd,
        "CLM_DRG_CD": clm_drg_cd,
        "ICD9_PRCDR_CD_1": icd9_prcdr_cd_1
    }

    with st.spinner("Analyzing claim..."):
        try:
            # Call the FastAPI backend
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()["prediction"]
                
                # Display results with visual cues
                if "Anomaly" in result or "Review" in result:
                    st.error(f"⚠️ **Result:** {result}")
                    st.warning("This claim exhibits anomalous structural or attributive patterns. Manual review is recommended.")
                else:
                    st.success(f"✅ **Result:** {result}")
                    st.info("This claim falls within the normal operational distribution.")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Ensure your FastAPI backend is running on http://localhost:8000")