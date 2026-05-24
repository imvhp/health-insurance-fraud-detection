# 📊 POST-TRAINING WORKFLOW: Complete Architecture

## Overview

After `run_pipeline.py` finishes training, you have a **production-ready ML model**. Now your **partner needs to integrate it** into their web application. This guide explains the complete flow.

---

## 🎯 What run_pipeline.py Produces

```
✅ Trained Isolation Forest model (saved in MLflow)
✅ Metrics dashboard (MLflow UI)
✅ Processed dataset (for reference)
```

**Location:**
- Model artifacts: `mlruns/0/models/.../artifacts/model`
- Metrics viewable at: `http://localhost:5000` (after running `mlflow ui`)

---

## 🔄 Complete System Architecture (After Training)

```
┌─────────────────────────────────────┐
│   PARTNER'S WEB APPLICATION         │
│   (Flask/FastAPI/Django)            │
│   - Admin Dashboard                 │
│   - User Interface                  │
│   - Form for entering claims        │
└──────────────┬──────────────────────┘
               │
               ↓ HTTP Request
               │ (claim data)
┌──────────────────────────────────────┐
│  YOUR PYTHON API (FastAPI)           │ ← You build this
│  Endpoints:                          │
│  - POST /predict                     │
│  - GET /model/status                 │
│  - GET /metrics                      │
└──────────────┬──────────────────────┘
               │
               ↓ (loads model)
┌──────────────────────────────────────┐
│  inference.py                        │
│  - Load model from MLflow            │
│  - Apply same transformations        │
│  - Run model.predict()               │
└──────────────┬──────────────────────┘
               │
               ↓ (anomaly scores)
┌──────────────────────────────────────┐
│  Business Logic Modules              │
│  - risk_scoring.py     (0-100%)      │
│  - alert_logic.py      (create alert?)│
│  - explainability.py   (why flagged?)│
│  - provider_service.py (stats)       │
└──────────────┬──────────────────────┘
               │
               ↓ JSON Response
               │
┌──────────────────────────────────────┐
│   PARTNER'S DATABASE                 │ ← Partner manages
│   - Store claims history             │
│   - Store alerts                     │
│   - Store investigator decisions     │
└──────────────────────────────────────┘
```

---

## 📋 Step-by-Step Flow: What Happens

### **Step 1: Partner Gets Claims Data (From Their Database)**

```python
# Partner's code (in their web app)
from src.serving.inference import predict
from src.models.risk_scoring import calculate_provider_risk_score
from src.models.alert_logic import should_create_alert

# Partner queries their database
claim = {
    "PRVDR_NUM": "1002CR",
    "NCH_PRMRY_PYR_CLM_PD_AMT": 25000,
    "AT_PHYSN_NPI": "1234567890",
    "OP_PHYSN_NPI": "9876543210",
    "OT_PHYSN_NPI": "UNKNOWN",
    "CLM_UTLZTN_DAY_CNT": 10,
    "ADMTNG_ICD9_DGNS_CD": "78650",
    "CLM_DRG_CD": "065",
    "ICD9_PRCDR_CD_1": "3991"
}
```

### **Step 2: Partner Calls YOUR API Endpoint**

```python
# Partner's web app calls your API
import requests

response = requests.post(
    "http://your-api-server:8000/predict",
    json=claim
)

result = response.json()
# Result contains:
# {
#   "anomaly_score": 0.75,  # 0-100%
#   "prediction": "ANOMALY",
#   "risk_category": "HIGH",
#   "confidence": 0.92
# }
```

### **Step 3: You Process The Claim (Your API)**

```python
# src/app/api.py (Your API endpoint)
from fastapi import FastAPI
from src.serving.inference import predict

@app.post("/predict")
def predict_claim(claim: ClaimData):
    # 1. Transform data (apply same encoding as training)
    anomaly_score = predict(claim.dict())
    
    # 2. Convert to human-readable format
    risk_pct = (anomaly_score + 1) / 2 * 100
    
    # 3. Apply business rules
    is_alert = should_create_alert(risk_pct)
    
    return {
        "anomaly_score": anomaly_score,
        "risk_percentage": risk_pct,
        "should_alert": is_alert,
        "timestamp": datetime.now()
    }
```

### **Step 4: Partner Stores Results in Their Database**

```python
# Partner's code
if result['should_alert']:
    # Create alert in their database
    db.create_alert(
        provider_id=claim['PRVDR_NUM'],
        risk_score=result['risk_percentage'],
        claim_data=claim,
        status='OPEN',
        created_at=datetime.now()
    )
    
    # Notify investigator
    send_notification_to_team(
        message=f"High-risk claim detected: {result['risk_percentage']}%"
    )
else:
    # Auto-approve low-risk claims
    db.approve_claim(claim_id=claim['id'])
```

---

## 🗄️ Database Setup (What Your Partner Needs)

Your partner should create tables like:

### **Claims Table**
```sql
CREATE TABLE claims (
    id INT PRIMARY KEY,
    provider_id VARCHAR(50),
    claim_amount FLOAT,
    patient_id INT,
    admission_date DATE,
    created_at TIMESTAMP
);
```

### **Predictions Table**
```sql
CREATE TABLE fraud_predictions (
    id INT PRIMARY KEY,
    claim_id INT,
    anomaly_score FLOAT,
    risk_percentage FLOAT,
    prediction VARCHAR(50),  -- 'NORMAL' or 'ANOMALY'
    created_at TIMESTAMP,
    FOREIGN KEY(claim_id) REFERENCES claims(id)
);
```

### **Alerts Table**
```sql
CREATE TABLE alerts (
    id INT PRIMARY KEY,
    provider_id VARCHAR(50),
    claim_id INT,
    risk_score FLOAT,
    status VARCHAR(50),  -- 'OPEN', 'CLOSED', 'APPROVED', 'DENIED'
    investigator_id INT,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    decision VARCHAR(50)
);
```

---

## 🚀 How to Set Up YOUR API

### **File: src/app/api.py** (Create/Update)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow.pyfunc
from datetime import datetime

app = FastAPI(title="Medicare Fraud Detection API")

# Load model at startup
MODEL = mlflow.pyfunc.load_model("path/to/model")

# Request schema (matches 9 baseline features)
class ClaimData(BaseModel):
    PRVDR_NUM: str
    NCH_PRMRY_PYR_CLM_PD_AMT: float
    AT_PHYSN_NPI: str
    OP_PHYSN_NPI: str
    OT_PHYSN_NPI: str
    CLM_UTLZTN_DAY_CNT: int
    ADMTNG_ICD9_DGNS_CD: str
    CLM_DRG_CD: str
    ICD9_PRCDR_CD_1: str

# Response schema
class PredictionResult(BaseModel):
    anomaly_score: float
    risk_percentage: float
    prediction: str  # "NORMAL" or "ANOMALY"
    should_alert: bool
    timestamp: datetime

# ============ ENDPOINTS ============

@app.get("/")
def root():
    return {"status": "ok", "service": "Medicare Fraud Detection API"}

@app.post("/predict")
def predict_claim(claim: ClaimData) -> PredictionResult:
    """
    Predict if a single claim is fraudulent.
    
    Input: 9 features for Isolation Forest
    Output: Risk score + alert decision
    """
    try:
        # Convert request to DataFrame (required by model)
        import pandas as pd
        df = pd.DataFrame([claim.dict()])
        
        # Get prediction from model
        anomaly_score = MODEL.predict(df)[0]
        
        # Convert to percentage
        risk_pct = (anomaly_score + 1) / 2 * 100
        
        # Determine if should alert
        should_alert = risk_pct > 50  # Your threshold
        
        return PredictionResult(
            anomaly_score=anomaly_score,
            risk_percentage=risk_pct,
            prediction="ANOMALY" if risk_pct > 50 else "NORMAL",
            should_alert=should_alert,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(claims: list[ClaimData]):
    """
    Predict multiple claims at once (more efficient).
    """
    results = []
    for claim in claims:
        result = predict_claim(claim)
        results.append(result)
    return results

@app.get("/model/status")
def model_status():
    """
    Check if model is loaded and working.
    """
    return {
        "status": "ready",
        "model": "Isolation Forest",
        "features": 9,
        "location": "MLflow artifacts"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "healthy"}
```

### **Run the API:**

```bash
pip install fastapi uvicorn

# Terminal 1: Start API server
cd e:\SCHOOL\Health\ Insurance\ Fraud\ Detection
python -m uvicorn src.app.api:app --reload --port 8000

# Terminal 2: Your partner calls it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PRVDR_NUM": "1002CR",
    "NCH_PRMRY_PYR_CLM_PD_AMT": 25000,
    ...
  }'
```

---

## 📚 What example_workflow.py Does

**example_workflow.py is a TEMPLATE/DEMO** showing:
- How to load predictions
- How to use business logic modules
- How to create alerts
- What the partner's code might look like

**It creates sample data because it assumes:**
```python
# Real: Partner loads from database
claims_df = db.query("SELECT * FROM claims WHERE created_today = TRUE")

# Demo: example_workflow.py generates sample data
claims_df, anomaly_scores = load_sample_data()
```

**To run the demo:**
```bash
python scripts/example_workflow.py
```

This shows output like:
```
Provider Risk Score: 65.0%
Risk Category: MEDIUM
Should create alert? True
Alert Priority: MEDIUM
```

---

## 🔗 Integration Summary: Step-by-Step

### **For YOU (ML Engineer):**

1. ✅ **Run pipeline training**
   ```bash
   python scripts/run_pipeline.py --input data/raw/...
   ```

2. ✅ **Start MLflow dashboard** (to see metrics)
   ```bash
   mlflow ui
   ```

3. ✅ **Build API server** (copy code from above to `src/app/api.py`)

4. ✅ **Start API server**
   ```bash
   python -m uvicorn src.app.api:app --reload --port 8000
   ```

5. ✅ **Test it locally**
   ```bash
   python scripts/example_workflow.py
   ```

### **For YOUR PARTNER (Web Developer):**

1. **Set up database** (PostgreSQL/MySQL/MongoDB)
   - Store claims
   - Store predictions
   - Store alerts

2. **Build web frontend** (React/Vue/Angular)
   - Form to submit claims
   - Display risk scores
   - Alert management dashboard

3. **Connect to your API**
   ```javascript
   // JavaScript example
   const response = await fetch('http://your-api:8000/predict', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(claimData)
   });
   const result = await response.json();
   ```

4. **Store results in their database**
   ```python
   db.create_prediction(
       claim_id=claim_id,
       anomaly_score=result['anomaly_score'],
       risk_percentage=result['risk_percentage']
   )
   ```

---

## 📊 Data Flow Summary

```
PARTNER'S APP                  YOUR SYSTEM                      PARTNER'S DB
┌──────────────┐              ┌──────────────┐                ┌──────────────┐
│ Web Form     │              │              │                │              │
│ (claim data) │──────────────→ FastAPI      │                │ Claims Table │
└──────────────┘              │ /predict     │                │ Predictions  │
                              │              │                │ Alerts       │
                              │ inference.py │                │              │
                              │ + models     │                │              │
                              │              │                │              │
                              └──────────────→──────────────→ Store results │
                                   JSON                       └──────────────┘
                                  response
```

---

## ✅ Testing Checklist

- [ ] Run `python scripts/run_pipeline.py` → Model trained ✓
- [ ] View MLflow dashboard → See metrics ✓
- [ ] Run `python -m uvicorn src.app.api:app --reload` → API starts ✓
- [ ] Test `/predict` endpoint → Returns risk score ✓
- [ ] Run `python scripts/example_workflow.py` → Shows complete flow ✓
- [ ] Partner creates database → Schema ready ✓
- [ ] Partner builds web form → Submits to your API ✓
- [ ] End-to-end test → Claim → Prediction → Alert ✓

---

## 🎓 Key Concepts

| Term | What It Is | Your Job | Partner's Job |
|------|-----------|----------|---------------|
| **Run Pipeline** | Train & save model | Execute this | Wait for completion |
| **MLflow** | Model tracking/storage | Use this | Just know it exists |
| **API Endpoint** | HTTP interface | Build this | Call this |
| **Database** | Store data + predictions | Optional | **MUST build this** |
| **Frontend** | User interface | (optional) | **MUST build this** |
| **Inference** | Making predictions | Your API handles | Via API call |

---

## 🚀 Next Steps

1. **Fix the MLflow URI issue** in `run_pipeline.py` → Use simple path (not file://)
2. **Run pipeline** → Train model
3. **Build API** → Copy code from api.py example above
4. **Share with partner** → They build database + frontend
5. **Test end-to-end** → Claim flows through system
6. **Deploy** → Put API on cloud server (AWS/Azure)

---

## 📞 Partner Integration Checklist

**Send this to your web partner:**

```
PARTNER TODO:
1. Create database (PostgreSQL recommended)
   - claims table
   - fraud_predictions table
   - alerts table
   
2. Build web form for claim submission
   
3. Call ML API endpoint:
   POST http://your-server:8000/predict
   
4. Store prediction results in database
   
5. Display results in dashboard
   
6. Create alert management system
```

---

## 💡 Important Notes

- **example_workflow.py** = Demo/template (not production code)
- **Your API** = Production code (partner calls this)
- **Partner's DB** = They build and manage this
- **Your job ends** = When API returns results
- **Partner's job starts** = How to use results in their app
