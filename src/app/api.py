"""
FASTAPI SERVING APPLICATION - Backend API for CMS Fraud Detection (PTL)
========================================================================

This API serves predictions from the dynamically selected anomaly detection model.

Run with:
    python -m uvicorn src.app.api:app --reload --port 8000

Test endpoints:
    GET  http://localhost:8000/
    GET  http://localhost:8000/health
    GET  http://localhost:8000/model/status
    POST http://localhost:8000/predict
    POST http://localhost:8000/predict_batch
    POST http://localhost:8000/explain
"""

import warnings
import sys
import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
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
from src.serving.shap_explainer import explain_claim
from src.risk_scoring import calculate_provider_risk_score
from src.alert_logic import should_create_alert

app = FastAPI(
    title="CMS Fraud Detection API (PTL)",
    description="ML serving API for detecting fraudulent claims with dynamic model selection",
    version="1.0.0"
)

# Mount static files for outputs
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
if os.path.exists(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

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
    model_version: str = Field("unknown", description="Model version (retrain timestamp or 'baseline')")
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
    model_version: str = "unknown"
    features_count: int
    ready: bool
    loaded_models: List[str]
    timestamp: datetime

class RetrainHistoryItem(BaseModel):
    version: str
    timestamp: str
    model_directory: str
    roc_curves_url: Optional[str]
    pr_curves_url: Optional[str]
    confusion_matrices_url: Optional[str]
    score_distributions_url: Optional[str]

class ModelHistoryResponse(BaseModel):
    baseline_version: str
    retrains: List[RetrainHistoryItem]

class RetrainRequest(BaseModel):
    """Request schema for triggering a model retrain"""
    since: Optional[str] = Field(None, description="Export claims since this date/duration (e.g. '7d', '2026-06-01')")
    days: Optional[int] = Field(4, description="Number of days of new data to process for retraining")

class RetrainResponse(BaseModel):
    """Response schema for retrain endpoint"""
    status: str
    message: str

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
        from src.serving.inference import get_model_version
        total_features = len(numeric_cols) + len(categorical_cols)
        return ModelStatusResponse(
            status="ready",
            model_directory=model_dir,
            model_version=get_model_version(model_dir),
            features_count=total_features,
            ready=True,
            loaded_models=list(models.keys()),
            timestamp=datetime.now()
        )
    except Exception as e:
        return ModelStatusResponse(
            status=f"error: {str(e)}",
            model_directory="N/A",
            model_version="unknown",
            features_count=0,
            ready=False,
            loaded_models=[],
            timestamp=datetime.now()
        )

@app.get("/model/history", response_model=ModelHistoryResponse)
def model_history():
    """Get history of retrained models and their evaluation image URLs."""
    from src.serving.inference import get_model_version
    
    retrains = []
    retrain_root = os.path.join(PROJECT_ROOT, "outputs", "retrain")
    
    if os.path.exists(retrain_root):
        # Sort chronologically
        subdirs = sorted(
            [d for d in os.listdir(retrain_root)
             if os.path.isdir(os.path.join(retrain_root, d))]
        )
        
        for d in subdirs:
            # Reconstruct the model dir path to get version
            model_dir = os.path.join(PROJECT_ROOT, "models", "retrain", d)
            version = get_model_version(model_dir)
            
            # Check for images in outputs
            out_dir = os.path.join(retrain_root, d)
            
            def get_url(filename):
                if os.path.exists(os.path.join(out_dir, filename)):
                    return f"/outputs/retrain/{d}/{filename}"
                return None
                
            retrains.append(RetrainHistoryItem(
                version=version,
                timestamp=d,
                model_directory=model_dir,
                roc_curves_url=get_url("roc_curves.png"),
                pr_curves_url=get_url("pr_curves.png"),
                confusion_matrices_url=get_url("confusion_matrices.png"),
                score_distributions_url=get_url("score_distributions.png")
            ))
            
    return ModelHistoryResponse(
        baseline_version="1.0.0",
        retrains=retrains
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
        model_version = res.get("model_version", "unknown")
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
            model_version=model_version,
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
        model_version = res.get("model_version", "unknown")
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
                model_version=model_version,
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
    Generate a SHAP-powered explanation for a single claim prediction.

    The endpoint re-uses the inference pipeline to obtain the preprocessed
    feature vector and then delegates to shap_explainer.explain_claim().
    Falls back to a heuristic explanation when SHAP is unavailable.
    """
    try:
        import numpy as np

        claim_dict = claim.dict()

        # ── 1. Load serving artifacts (cached after first call) ──────────────
        preprocessor, models, numeric_cols, categorical_cols, _ = get_serving_artifacts()

        # ── 2. Run inference to get risk score + selected model ──────────────
        res = predict_claims([claim_dict])
        pred = res["predictions"][0]
        risk_percentage = pred["risk_percentage"]
        model_name      = pred["model_selected"]
        provider_id     = pred["provider_id"]
        clf             = models[model_name]

        # ── 3. Build the preprocessed feature vector for SHAP ────────────────
        import pandas as pd
        from features import add_domain_features

        df        = pd.DataFrame([claim_dict])
        df_feats  = add_domain_features(df)

        # Align columns exactly as the preprocessor expects
        expected_cols = list(numeric_cols) + list(categorical_cols)
        for col in expected_cols:
            if col not in df_feats.columns:
                df_feats[col] = np.nan

        X_df         = df_feats[expected_cols]
        X_transformed = preprocessor.transform(X_df)          # (1, n_features)
        x_row         = np.asarray(X_transformed[0]).ravel()  # 1-D

        # Feature names after preprocessing (numeric first, then categorical)
        try:
            feature_names = (
                list(preprocessor.transformers_[0][1].get_feature_names_out(numeric_cols))
                + list(preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols))
            )
        except Exception:
            # Fallback: raw column names (no one-hot expansion info)
            feature_names = expected_cols

        # ── 4. Delegate to SHAP explainer ────────────────────────────────────
        explanation = explain_claim(
            model_name=model_name,
            clf=clf,
            x_row=x_row,
            feature_names=feature_names,
            risk_percentage=risk_percentage,
            provider_id=provider_id,
        )

        return explanation

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )

# ============ MANAGEMENT ENDPOINTS ============

@app.post("/retrain", response_model=RetrainResponse)
def trigger_retrain(background_tasks: BackgroundTasks, req: Optional[RetrainRequest] = None):
    """
    Trigger the retrain pipeline in the background.
    This will:
    1. Export new claims from the database
    2. Run the retraining cycle with the new data
    """
    if req is None:
        req = RetrainRequest()
        
    def run_retrain_pipeline(since: Optional[str], days: int):
        import subprocess
        print(f"Starting retrain pipeline background task (since={since}, days={days})")
        
        # Step 1: Export
        export_cmd = [sys.executable, "src/export_claims_csv.py"]
        if since:
            export_cmd.extend(["--since", since])
            
        try:
            print(f"Running export: {' '.join(export_cmd)}")
            subprocess.run(export_cmd, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
            # Continuing anyway as per bat script
            
        # Step 2: Retrain
        retrain_cmd = [sys.executable, "src/retrain_cycle.py", "--new-data-dir", "data/new", "--days", str(days)]
        try:
            print(f"Running retrain: {' '.join(retrain_cmd)}")
            subprocess.run(retrain_cmd, cwd=PROJECT_ROOT, check=True)
            print("Retrain pipeline completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Retrain cycle failed: {e}")

    background_tasks.add_task(run_retrain_pipeline, req.since, req.days)
    
    return RetrainResponse(
        status="accepted",
        message="Retrain pipeline started in background. Check server logs for progress."
    )

