"""
FASTAPI SERVING APPLICATION - Backend API for CMS Fraud Detection (PTL)
========================================================================

This API serves predictions from the dynamically selected anomaly detection model.

Run with:
    python -m uvicorn src.app.api:app --reload --port 8002

Test endpoints:
    GET  http://localhost:8002/
    GET  http://localhost:8002/health
    GET  http://localhost:8002/model/status
    POST http://localhost:8002/predict
    POST http://localhost:8002/predict_batch
    POST http://localhost:8002/explain
"""

import warnings
import sys
import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Suppress sklearn imputer warning for columns with no observed values during single-claim inference
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning
)

# Setup path imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import serving and logic modules
from src.serving.inference import predict_claims, get_serving_artifacts
from src.risk_scoring import calculate_provider_risk_score
from src.alert_logic import should_create_alert

app = FastAPI(
    title="CMS Fraud Detection API (PTL)",
    description="ML serving API for detecting fraudulent claims with dynamic model selection",
    version="1.0.0"
)

# ============ REQUEST/RESPONSE SCHEMAS ============

class ClaimData(BaseModel):
    """
    Schema representing claims features.
    Features mapped to match fields used in the CMS Synthetic Claims project.
    All fields are made optional with sensible defaults for maximum API consistency.
    """
    DESYNPUF_ID: Optional[str] = Field(None, description="Beneficiary ID")
    CLM_ID: Optional[str] = Field(None, description="Claim ID")
    SEGMENT: Optional[int] = Field(None, description="Segment")
    CLM_FROM_DT: Optional[int] = Field(None, description="Claim start date (YYYYMMDD)")
    CLM_THRU_DT: Optional[int] = Field(None, description="Claim end date (YYYYMMDD)")
    PRVDR_NUM: Optional[str] = Field("UNKNOWN", description="Provider ID")
    CLM_PMT_AMT: Optional[float] = Field(0.0, description="Claim payment amount")
    NCH_PRMRY_PYR_CLM_PD_AMT: Optional[float] = Field(0.0, description="Primary payer claim paid amount")
    AT_PHYSN_NPI: Optional[str] = Field("UNKNOWN", description="Attending physician NPI")
    OP_PHYSN_NPI: Optional[str] = Field("UNKNOWN", description="Operating physician NPI")
    OT_PHYSN_NPI: Optional[str] = Field("UNKNOWN", description="Other physician NPI")
    CLM_ADMSN_DT: Optional[int] = Field(None, description="Admission date (YYYYMMDD)")
    ADMTNG_ICD9_DGNS_CD: Optional[str] = Field("UNKNOWN", description="Admitting ICD9 diagnosis code")
    CLM_PASS_THRU_PER_DIEM_AMT: Optional[float] = Field(0.0, description="Claim pass through per diem amount")
    NCH_BENE_IP_DDCTBL_AMT: Optional[float] = Field(0.0, description="IP deductible amount")
    NCH_BENE_PTA_COINSRNC_LBLTY_AM: Optional[float] = Field(0.0, description="IP coinsurance liability amount")
    NCH_BENE_BLOOD_DDCTBL_LBLTY_AM: Optional[float] = Field(0.0, description="Blood deductible liability amount")
    CLM_UTLZTN_DAY_CNT: Optional[int] = Field(0, description="Claim utilization day count")
    NCH_BENE_DSCHRG_DT: Optional[int] = Field(None, description="Discharge date (YYYYMMDD)")
    CLM_DRG_CD: Optional[str] = Field("UNKNOWN", description="Claim DRG code")
    ICD9_DGNS_CD_1: Optional[str] = Field(None, description="ICD9 diagnosis code 1")
    ICD9_PRCDR_CD_1: Optional[str] = Field(None, description="ICD9 procedure code 1")

class PredictionResponse(BaseModel):
    """Response from single prediction"""
    anomaly_score: float = Field(..., description="Raw model score (native units, model-dependent)")
    risk_percentage: float = Field(..., description="Risk score as percentage (0-100), derived from model-native boundary")
    prediction: str = Field(..., description="'NORMAL' (LOW) or 'ANOMALY' (MEDIUM or HIGH)")
    should_alert: bool = Field(..., description="Should create fraud alert? (risk_percentage > 70)")
    provider_id: str = Field(..., description="Provider ID from claim")
    model_selected: str = Field(..., description="Model chosen for this prediction")
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPredictionResponse(BaseModel):
    """Response from batch predictions"""
    predictions: List[PredictionResponse]
    total: int
    anomalies_detected: int
    alerts_triggered: int
    model_selected: str = Field(..., description="Dynamically selected model")

class ModelStatusResponse(BaseModel):
    """Response for model status check"""
    status: str
    model_directory: str
    features_count: int
    ready: bool
    loaded_models: List[str]
    timestamp: datetime

# ============ HEALTH CHECK ENDPOINTS ============

@app.get("/")
def root():
    """Root endpoint - basic status check"""
    return {
        "status": "ok",
        "service": "CMS Fraud Detection API (PTL)",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring/load balancing"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "CMS Fraud Detection (PTL)"
    }

@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    """Check if preprocessor and models are loaded and ready"""
    try:
        preprocessor, models, numeric_cols, categorical_cols, model_dir = get_serving_artifacts()
        total_features = len(numeric_cols) + len(categorical_cols)
        return ModelStatusResponse(
            status="ready",
            model_directory=model_dir,
            features_count=total_features,
            ready=True,
            loaded_models=list(models.keys()),
            timestamp=datetime.now()
        )
    except Exception as e:
        return ModelStatusResponse(
            status=f"error: {str(e)}",
            model_directory="N/A",
            features_count=0,
            ready=False,
            loaded_models=[],
            timestamp=datetime.now()
        )

# ============ PREDICTION ENDPOINTS ============

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(claim: ClaimData):
    """
    Predict if a single claim is fraudulent.
    Automatically profiles features to select the best model.
    """
    try:
        claim_dict = claim.dict()
        res = predict_claims([claim_dict])
        
        model_name = res["model_selected"]
        pred = res["predictions"][0]
        
        # Calculate if should alert
        should_alert = should_create_alert(pred["risk_percentage"])
        
        return PredictionResponse(
            anomaly_score=pred["anomaly_score"],
            risk_percentage=pred["risk_percentage"],
            prediction=pred["prediction"],
            should_alert=should_alert,
            provider_id=pred["provider_id"],
            model_selected=pred["model_selected"],
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(claims: List[ClaimData]):
    """
    Predict multiple claims at once.
    Profiles the batch features as a group to select the best model.
    """
    try:
        claims_dicts = [c.dict() for c in claims]
        res = predict_claims(claims_dicts)
        
        model_name = res["model_selected"]
        predictions = []
        alerts_count = 0
        anomaly_count = 0
        
        for pred in res["predictions"]:
            should_alert = should_create_alert(pred["risk_percentage"])
            
            response_pred = PredictionResponse(
                anomaly_score=pred["anomaly_score"],
                risk_percentage=pred["risk_percentage"],
                prediction=pred["prediction"],
                should_alert=should_alert,
                provider_id=pred["provider_id"],
                model_selected=pred["model_selected"],
                timestamp=datetime.now()
            )
            predictions.append(response_pred)
            
            if should_alert:
                alerts_count += 1
            if pred["prediction"] == "ANOMALY":
                anomaly_count += 1
                
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(claims),
            anomalies_detected=anomaly_count,
            alerts_triggered=alerts_count,
            model_selected=model_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/explain")
def explain_prediction(claim: ClaimData):
    """
    Generate explanation for a prediction.
    Uses rule-based logic to trace risk factors.
    """
    try:
        claim_dict = claim.dict()
        res = predict_claims([claim_dict])

        pred = res["predictions"][0]
        risk_percentage = pred["risk_percentage"]
        model_name = pred["model_selected"]
        is_anomaly = risk_percentage > 50.0

        if not is_anomaly:
            return {
                "top_factors": [],
                "summary": "Claim falls within normal distribution parameters. Not flagged.",
                "confidence": round((100 - risk_percentage) / 100, 2),
                "method": f"Heuristic / {model_name}",
                "provider_id": pred["provider_id"]
            }

        # Build rule-based explanation for anomaly
        top_factors = []
        if claim.CLM_PMT_AMT is not None and claim.CLM_PMT_AMT > 5000:
            # Calculate impact dynamically based on amount over 5000
            base_impact = 0.20
            dynamic_impact = min(0.60, round(base_impact + ((claim.CLM_PMT_AMT - 5000) / 100000.0), 2))
            top_factors.append({
                "feature": "CLM_PMT_AMT",
                "impact": dynamic_impact,
                "direction": "high",
                "value": claim.CLM_PMT_AMT
            })
            
        if claim.CLM_UTLZTN_DAY_CNT is not None and claim.CLM_UTLZTN_DAY_CNT > 7:
            # Calculate impact dynamically based on days over 7
            base_impact = 0.15
            dynamic_impact = min(0.50, round(base_impact + ((claim.CLM_UTLZTN_DAY_CNT - 7) / 50.0), 2))
            top_factors.append({
                "feature": "CLM_UTLZTN_DAY_CNT",
                "impact": dynamic_impact,
                "direction": "high",
                "value": claim.CLM_UTLZTN_DAY_CNT
            })
            
        # Sort factors by impact
        top_factors.sort(key=lambda x: x["impact"], reverse=True)

        summary = f"CLAIM FLAGGED ({risk_percentage:.1f}% Risk). "
        if top_factors:
            summary += "Top flagged factors: " + ", ".join(
                [f"{f['feature']} ({f['value']})" for f in top_factors]
            )
        else:
            summary += f"Identified as anomaly by {model_name} feature distribution."

        return {
            "top_factors": top_factors,
            "summary": summary,
            "confidence": round(risk_percentage / 100, 2),
            "method": f"Heuristic / {model_name}",
            "provider_id": pred["provider_id"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )

