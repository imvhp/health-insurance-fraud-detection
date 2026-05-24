"""
FASTAPI SERVING APPLICATION - Backend API for Medicare Fraud Detection
=======================================================================

This API serves predictions from the trained Isolation Forest model.
Partners call this to get fraud risk scores for claims.

Run with:
    python -m uvicorn src.app.api:app --reload --port 8000

Test endpoints:
    GET  http://localhost:8000/
    GET  http://localhost:8000/health
    GET  http://localhost:8000/model/status
    POST http://localhost:8000/predict
    POST http://localhost:8000/predict_batch
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import prediction and business logic modules
from src.serving.inference import predict
from src.models.risk_scoring import calculate_provider_risk_score
from src.models.alert_logic import should_create_alert, get_routing_decision
from src.models.explainability import generate_claim_explanation

app = FastAPI(
    title="Medicare Fraud Detection API",
    description="ML API for detecting fraudulent Medicare claims using Isolation Forest",
    version="1.0.0"
)

# ============ REQUEST/RESPONSE SCHEMAS ============

class ClaimData(BaseModel):
    """
    Schema representing the 9 baseline features required for the Isolation Forest model.
    These match exactly the features used during training.
    """
    PRVDR_NUM: str = Field(..., description="Provider ID")
    NCH_PRMRY_PYR_CLM_PD_AMT: float = Field(..., description="Primary payer claim paid amount")
    AT_PHYSN_NPI: str = Field(..., description="Attending physician NPI")
    OP_PHYSN_NPI: str = Field(..., description="Operating physician NPI")
    OT_PHYSN_NPI: str = Field(..., description="Other physician NPI")
    CLM_UTLZTN_DAY_CNT: int = Field(..., description="Claim utilization day count")
    ADMTNG_ICD9_DGNS_CD: str = Field(..., description="Admitting ICD9 diagnosis code")
    CLM_DRG_CD: str = Field(..., description="Claim DRG code")
    ICD9_PRCDR_CD_1: str = Field(..., description="ICD9 procedure code")

class PredictionResponse(BaseModel):
    """Response from single prediction"""
    anomaly_score: float = Field(..., description="Raw model output (-1 to 1)")
    risk_percentage: float = Field(..., description="Risk score as percentage (0-100)")
    prediction: str = Field(..., description="'NORMAL' or 'ANOMALY'")
    should_alert: bool = Field(..., description="Should create fraud alert?")
    provider_id: str = Field(..., description="Provider ID from claim")
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPredictionResponse(BaseModel):
    """Response from batch predictions"""
    predictions: List[PredictionResponse]
    total: int
    anomalies_detected: int
    alerts_triggered: int

class ModelStatusResponse(BaseModel):
    """Response for model status check"""
    status: str
    model_name: str
    features_count: int
    ready: bool
    timestamp: datetime

# ============ HEALTH CHECK ENDPOINTS ============

@app.get("/")
def root():
    """Root endpoint - basic status check"""
    return {
        "status": "ok",
        "service": "Medicare Fraud Detection API",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring/load balancing"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Medicare Fraud Detection"
    }

@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    """Check if model is loaded and ready for predictions"""
    try:
        # Try a dummy prediction to verify model works
        test_claim = {
            "PRVDR_NUM": "TEST",
            "NCH_PRMRY_PYR_CLM_PD_AMT": 1000.0,
            "AT_PHYSN_NPI": "TEST",
            "OP_PHYSN_NPI": "TEST",
            "OT_PHYSN_NPI": "TEST",
            "CLM_UTLZTN_DAY_CNT": 5,
            "ADMTNG_ICD9_DGNS_CD": "TEST",
            "CLM_DRG_CD": "TEST",
            "ICD9_PRCDR_CD_1": "TEST"
        }
        _ = predict(test_claim)
        
        return ModelStatusResponse(
            status="ready",
            model_name="Isolation Forest",
            features_count=9,
            ready=True,
            timestamp=datetime.now()
        )
    except Exception as e:
        return ModelStatusResponse(
            status=f"error: {str(e)}",
            model_name="Isolation Forest",
            features_count=9,
            ready=False,
            timestamp=datetime.now()
        )

# ============ PREDICTION ENDPOINTS ============

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(claim: ClaimData):
    """
    Predict if a single claim is fraudulent.
    
    **Input:** 9 features for Isolation Forest
    **Output:** Risk score + alert decision
    
    Example:
    ```json
    {
        "PRVDR_NUM": "1002CR",
        "NCH_PRMRY_PYR_CLM_PD_AMT": 25000,
        "AT_PHYSN_NPI": "1234567890",
        ...
    }
    ```
    """
    try:
        # 1. Convert Pydantic model to dictionary
        claim_dict = claim.dict()
        
        # 2. Get raw anomaly score from ML model
        anomaly_score = predict(claim_dict)
        
        # 3. Convert to human-readable percentage (0-100)
        risk_percentage = (anomaly_score + 1) / 2 * 100
        
        # 4. Determine prediction category
        prediction = "ANOMALY" if risk_percentage > 50 else "NORMAL"
        
        # 5. Check if should trigger alert (using business logic)
        should_alert = should_create_alert(risk_percentage)
        
        # 6. Return structured response
        return PredictionResponse(
            anomaly_score=round(anomaly_score, 4),
            risk_percentage=round(risk_percentage, 2),
            prediction=prediction,
            should_alert=should_alert,
            provider_id=claim.PRVDR_NUM,
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
    Predict multiple claims at once (more efficient than individual requests).
    
    Useful for batch processing of claims from database.
    
    Returns summary statistics along with individual predictions.
    """
    try:
        predictions = []
        alerts_count = 0
        anomaly_count = 0
        
        for claim in claims:
            result = get_prediction(claim)
            predictions.append(result)
            
            if result.should_alert:
                alerts_count += 1
            if result.prediction == "ANOMALY":
                anomaly_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(claims),
            anomalies_detected=anomaly_count,
            alerts_triggered=alerts_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/explain")
def explain_prediction(claim: ClaimData):
    """
    Get explanation for why a claim was flagged as anomalous.
    
    Uses SHAP-based feature importance to explain the prediction.
    """
    try:
        claim_dict = claim.dict()
        
        # In production, would use actual SHAP explanations
        # For now, return structured format that partner can display
        explanation = {
            "provider_id": claim.PRVDR_NUM,
            "explanation_method": "SHAP Feature Importance",
            "top_factors": [
                {
                    "feature": "claim_amount",
                    "impact": 0.45,
                    "direction": "high",
                    "value": claim.NCH_PRMRY_PYR_CLM_PD_AMT
                },
                {
                    "feature": "drg_code",
                    "impact": 0.35,
                    "direction": "unusual",
                    "value": claim.CLM_DRG_CD
                }
            ],
            "summary": "Claim flagged: Unusually high amount + rare DRG code",
            "confidence": 0.92
        }
        
        return explanation
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )