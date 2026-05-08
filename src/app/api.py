"""
FASTAPI SERVING APPLICATION - Backend API
=========================================
Run this with: python -m uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the actual prediction pipeline from your serving module
from src.serving.inference import predict 

app = FastAPI(
    title="Medicare Fraud Detection API",
    description="ML API for detecting fraudulent Medicare claims using Isolation Forest",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Medicare Fraud Detection API is running"}

# === REQUEST DATA SCHEMA ===
class ClaimData(BaseModel):
    """
    Schema representing the 9 baseline features required for the Isolation Forest model.
    """
    PRVDR_NUM: str
    NCH_PRMRY_PYR_CLM_PD_AMT: float
    AT_PHYSN_NPI: str
    OP_PHYSN_NPI: str
    OT_PHYSN_NPI: str
    CLM_UTLZTN_DAY_CNT: int
    ADMTNG_ICD9_DGNS_CD: str
    CLM_DRG_CD: str
    ICD9_PRCDR_CD_1: str

@app.post("/predict")
def get_prediction(data: ClaimData):
    try:
        # 1. Convert the validated Pydantic model to a standard Python dictionary
        claim_dict = data.dict()
        
        # 2. Pass the dictionary directly to our inference pipeline
        # The predict() function automatically handles transformations and string formatting
        result = predict(claim_dict)
        
        # 3. Return the JSON response to the Streamlit frontend
        return {"prediction": result}
        
    except Exception as e:
        # If anything fails in inference (like a missing model), return a clean HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))