# 📊 COMPLETE SYSTEM ARCHITECTURE & DATA FLOW

## ✅ Summary: What Happens After run_pipeline.py

```
════════════════════════════════════════════════════════════════════════════════════

AFTER run_pipeline.py FINISHES:
  ✅ Trained model saved in MLflow
  ✅ Metrics logged (Precision, Recall, F1, ROC-AUC)
  ✅ Processed data saved to data/processed/claims_processed.csv

════════════════════════════════════════════════════════════════════════════════════
```

---

## 🏗️ Your Role vs Partner's Role

### YOU (ML Engineer):
1. ✅ Train model (`run_pipeline.py`)
2. ✅ Build API server (`api.py`)
3. ✅ Provide prediction results (JSON)
4. ✅ Monitor model performance

### YOUR PARTNER (Web Developer):
1. ✅ Create database (PostgreSQL/MySQL)
2. ✅ Build web form for claims
3. ✅ Call YOUR API endpoint
4. ✅ Store predictions + create alerts
5. ✅ Display results to users

---

## 📈 Complete Data Flow After Training

```
PARTNER'S SYSTEM                           YOUR SYSTEM                    PARTNER'S DB
═════════════════════════════════════════════════════════════════════════════════════

┌──────────────────┐
│  USER SUBMITS    │
│  CLAIM FORM      │
│                  │
│  (Web Frontend)  │
└────────┬─────────┘
         │
         ↓ (claim data)
┌──────────────────┐                  ┌────────────────────┐
│  VALIDATE FORM   │                  │                    │
│  & PREPARE DATA  │                  │                    │
│                  │                  │                    │
│  (JavaScript)    │                  │                    │
└────────┬─────────┘                  │                    │
         │                            │                    │
         ↓ HTTP POST request          │                    │
    ┌────────────────────────────────→│  FASTAPI SERVER    │
    │  {                              │  (/predict)        │
    │    PRVDR_NUM: "1002CR",         │                    │
    │    CLM_PMT_AMT: 25000,          │  inference.py      │
    │    AT_PHYSN_NPI: "12345...",    │  ├─ Load model     │
    │    ...9 features                │  ├─ Transform data │
    │  }                              │  └─ Predict (IF)   │
    │                                 │                    │
    │  ← HTTP response                │  risk_scoring.py   │
    │  {                              │  ├─ Convert 0-100% │
    └───────────────────────────────→│  │                  │
         ↓                            │  alert_logic.py    │
    │ anomaly_score: -0.12,           │  └─ Should alert?  │
    │ risk_percentage: 43.8,          │                    │
    │ prediction: "NORMAL",           │  explainability.py │
    │ should_alert: false             │  ├─ SHAP features  │
    │}                                │  └─ Explanations   │
    │                                 │                    │
    └─────────────────────────────────│────────────────────┘
         ↓
┌──────────────────────┐              ┌────────────────────────┐
│  STORE IN DATABASE   │              │                        │
│                      │              │  (Optional logs)       │
│  INSERT predictions  │              │  - API call history    │
│  INSERT alerts       │              │  - Prediction accuracy │
│  UPDATE claim status │              │  - Error logs          │
└────────┬─────────────┘              └────────────────────────┘
         │
         ↓
┌──────────────────────┐
│  UPDATE DASHBOARD    │
│                      │
│  Show to user:       │
│  - Risk score        │
│  - Decision (OK/ALT) │
│  - Next steps        │
│  - Explanation       │
└──────────────────────┘
```

---

## 🔌 API Contract Between You & Partner

### What Partner Sends (Request)

```json
POST /predict
Content-Type: application/json

{
  "PRVDR_NUM": "1002CR",
  "NCH_PRMRY_PYR_CLM_PD_AMT": 25000.00,
  "AT_PHYSN_NPI": "1234567890",
  "OP_PHYSN_NPI": "9876543210",
  "OT_PHYSN_NPI": "5555555555",
  "CLM_UTLZTN_DAY_CNT": 10,
  "ADMTNG_ICD9_DGNS_CD": "78650",
  "CLM_DRG_CD": "065",
  "ICD9_PRCDR_CD_1": "3991"
}
```

### What You Return (Response)

```json
{
  "anomaly_score": -0.1234,
  "risk_percentage": 43.83,
  "prediction": "NORMAL",
  "should_alert": false,
  "provider_id": "1002CR",
  "timestamp": "2026-05-23T10:30:45.123456"
}
```

---

## 🗄️ Database Tables Partner Must Create

### Table 1: claims
Stores the claims data
```sql
CREATE TABLE claims (
    id INT PRIMARY KEY AUTO_INCREMENT,
    provider_id VARCHAR(50) NOT NULL,
    claim_amount DECIMAL(12,2),
    patient_id INT,
    admission_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table 2: fraud_predictions
Stores prediction results from your API
```sql
CREATE TABLE fraud_predictions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    claim_id INT NOT NULL,
    anomaly_score FLOAT,
    risk_percentage FLOAT,
    prediction VARCHAR(50),          -- 'NORMAL' or 'ANOMALY'
    should_alert BOOLEAN,
    api_call_time FLOAT,             -- How long API took
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(claim_id) REFERENCES claims(id)
);
```

### Table 3: alerts
Tracks alerts created from high-risk predictions
```sql
CREATE TABLE alerts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    claim_id INT NOT NULL,
    provider_id VARCHAR(50),
    risk_score FLOAT,
    status VARCHAR(50),              -- 'OPEN', 'CLOSED', 'APPROVED', 'DENIED'
    investigator_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    decision VARCHAR(100),           -- Why it was approved/denied
    FOREIGN KEY(claim_id) REFERENCES claims(id)
);
```

---

## 🚀 Complete Execution Timeline

### Day 1 - Model Training (You)
```
Morning:   python scripts/run_pipeline.py
           → Model trained ✓
           → Metrics logged ✓
           
Afternoon: mlflow ui
           → View results in dashboard ✓
```

### Day 1-2 - API Development (You)
```
           Update src/app/api.py
           → Already done! ✓
           
           python -m uvicorn src.app.api:app --reload --port 8000
           → API running ✓
           
           Test all endpoints ✓
```

### Day 2 - Share with Partner (You)
```
           Send them:
           ✓ API documentation (POST_TRAINING_ARCHITECTURE.md)
           ✓ Example request/response
           ✓ Database schema
           ✓ API URL
```

### Day 3+ - Partner Integration (Partner)
```
Day 3:     Create database tables ✓
           Build web form ✓
           
Day 4:     Test API calls from form ✓
           Store predictions ✓
           
Day 5:     Build alert dashboard ✓
           Display results to users ✓
           
Day 6+:    Monitor & improve ✓
           Track prediction accuracy ✓
```

---

## 📊 Monitoring After Deployment

### KPIs You Should Track

| Metric | How to Check | Acceptable Range |
|--------|-------------|------------------|
| **API Response Time** | `result.timestamp` - request_time | < 500ms |
| **Prediction Accuracy** | Compare with ground truth | > 85% |
| **Alert False Positive Rate** | Manual review | < 10% |
| **Model Drift** | Retrain metrics quarterly | Precision ±5% |
| **API Uptime** | Monitor 24/7 | > 99.5% |

### Actions If Things Break

```python
# If accuracy drops:
→ Run new training: python scripts/run_pipeline.py
→ Update model in MLflow
→ Partner's API auto-loads new version

# If API is slow:
→ Check model file size
→ Move to faster hardware
→ Add caching layer

# If false positive rate high:
→ Adjust risk threshold in alert_logic.py
→ Retune model contamination parameter
```

---

## 🎯 Where Each File Lives & What It Does

```
YOUR PROJECT                          PARTNER'S PROJECT
═══════════════════════════════════════════════════════════════════

src/
├── data/
│   ├── load_data.py            (Load CSV)
│   └── preprocess.py           (Clean data)
│
├── features/
│   └── build_features.py       (Select 9 features, encode)
│
├── models/
│   ├── train.py                (Train Isolation Forest)
│   ├── risk_scoring.py         ────→ Used by partner
│   ├── alert_logic.py          ────→ Used by partner
│   ├── explainability.py       ────→ Used by partner
│   └── provider_service.py     ────→ Used by partner
│
├── serving/
│   └── inference.py            (Load model + predict)
│
└── app/
    └── api.py                  (FastAPI endpoints)
            ↓
         [YOUR API SERVER]
            ↓
    ┌──────────────────────────────┐
    │  POST /predict               │
    │  POST /predict_batch         │
    │  GET /model/status           │
    │  POST /explain               │
    └──────────────────────────────┘
            ↓
         [PARTNER CALLS IT]
            ↓
    ┌──────────────────────────────┐
    │  Frontend (React/Vue)        │
    │  Backend (Flask/Django)      │
    │  Database (PostgreSQL)       │
    │  Dashboard (display results) │
    └──────────────────────────────┘
```

---

## 💡 Key Points to Understand

### 1. example_workflow.py = DEMO ONLY
- Shows how business logic works
- Creates fake data for testing
- Partner doesn't need this (they have their own data)
- You use this to test your modules before sending to partner

### 2. Your API = Single Source of Truth
- Partner calls YOUR API, not example_workflow.py
- API uses inference.py to load model
- API uses business logic modules (risk_scoring, alert_logic, etc.)
- API returns JSON that partner stores in their database

### 3. Partner's Database = Where Everything Ends Up
- You don't manage partner's database
- Partner stores your predictions in their database
- Partner creates alerts based on your risk scores
- Partner's users see results in their dashboard

### 4. MLflow = Model Versioning
- You train once, MLflow saves the model
- If you retrain, new version saved
- API automatically uses latest version
- Partner doesn't need to worry about versions

### 5. Batch Processing = Efficiency
- If partner has 1000 claims to check
- Send via `/predict_batch` endpoint (faster)
- Instead of 1000 individual requests
- You return 1000 results at once

---

## ✅ Deployment Checklist

- [ ] Model trained and metrics reviewed
- [ ] MLflow dashboard shows good performance
- [ ] API server starts without errors
- [ ] All endpoints respond (test with Swagger)
- [ ] Batch predictions work
- [ ] Example workflow runs successfully
- [ ] Documentation created (send to partner)
- [ ] Partner creates database schema
- [ ] Partner tests API integration
- [ ] End-to-end testing (real claims)
- [ ] Performance monitoring set up
- [ ] Alerting configured
- [ ] Production deployment completed

---

## 🎓 Architecture Summary in One Diagram

```
TRAINING PHASE                    SERVING PHASE
══════════════════════════════════════════════════════════════

Raw Data                          User clicks "Submit Claim"
    ↓                                 ↓
preprocess.py                    Partner's Web Form
    ↓                                 ↓
validate_data.py                 HTTP POST /predict
    ↓                                 ↓
build_features.py                YOUR API (api.py)
    ↓                                 ↓
IsolationForest.fit()            inference.py (load model)
    ↓                                 ↓
MLflow (save model)              Model.predict()
    ↓                                 ↓
run_pipeline.py                  risk_scoring.py
(DONE - Model ready)             alert_logic.py
                                     ↓
                                 JSON Response
                                     ↓
                                 Partner's DB
                                     ↓
                                 Partner's Dashboard
```

---

## 🌟 What You've Built

✅ **End-to-End ML Pipeline**
- From raw data → trained model
- Automated data cleaning
- Feature engineering
- Model training & evaluation

✅ **Production API**
- FastAPI with Swagger documentation
- Single prediction endpoint
- Batch prediction endpoint
- Model status monitoring
- Error handling

✅ **Business Logic Modules**
- Risk scoring (0-100%)
- Alert creation rules
- Explainability (SHAP)
- Provider statistics

✅ **Complete Documentation**
- Architecture guides
- Quick start guides
- Integration examples
- Database schemas

---

This is **production-ready**. Your partner can now build their web application on top of your API! 🚀
