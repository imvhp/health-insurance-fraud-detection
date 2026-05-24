# 🚀 COMPLETE QUICKSTART: From Training to Production

## Phase 1: Train Model (You Run This)

### Step 1: Run Training Pipeline

```bash
cd e:\SCHOOL\Health\ Insurance\ Fraud\ Detection

python scripts/run_pipeline.py \
  --input data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
  --target anomaly_label
```

**What happens:**
- ✅ Loads raw claims data
- ✅ Preprocesses & validates
- ✅ Injects synthetic anomalies (2%)
- ✅ Trains Isolation Forest
- ✅ Saves model to MLflow
- ✅ Saves processed data to `data/processed/claims_processed.csv`

**Time:** ~2-5 minutes

---

## Phase 2: View Training Results

### Step 2a: View MLflow Dashboard

```bash
# Terminal 1: Start MLflow UI
mlflow ui

# Then open browser: http://localhost:5000
```

**What you see:**
- Training metrics: Precision, Recall, F1, ROC-AUC
- Model artifacts location
- Hyperparameters used
- Training time

### Step 2b: View Processed Data

```bash
# Check the cleaned/processed dataset
python -c "import pandas as pd; df = pd.read_csv('data/processed/claims_processed.csv'); print(df.head()); print(df.shape)"
```

**Output:** 
- Dataset shape: (rows, 10) — 9 features + 1 label
- First few rows with all features encoded as numbers

---

## Phase 3: Start Your API Server

### Step 3: Run FastAPI Server

```bash
# Terminal 2: Start API server
python -m uvicorn src.app.api:app --reload --port 8000

# Output should show:
# Uvicorn running on http://127.0.0.1:8000
# Press CTRL+C to quit
```

**Available Endpoints:**

| Method | URL | Purpose |
|--------|-----|---------|
| GET | `http://localhost:8000/` | Status check |
| GET | `http://localhost:8000/health` | Health monitoring |
| GET | `http://localhost:8000/model/status` | Check if model ready |
| POST | `http://localhost:8000/predict` | Predict single claim |
| POST | `http://localhost:8000/predict_batch` | Predict multiple claims |
| POST | `http://localhost:8000/explain` | Explain prediction |

---

## Phase 4: Test Your API

### Step 4a: Quick Test (Browser/Postman)

Open browser to: `http://localhost:8000/docs`

**Swagger UI appears** - Interactive API documentation!

Click on **POST /predict** → **Try it out** → Copy-paste this valid JSON:

```json
{
  "PRVDR_NUM": "1002CR",
  "NCH_PRMRY_PYR_CLM_PD_AMT": 25000.0,
  "AT_PHYSN_NPI": "1234567890",
  "OP_PHYSN_NPI": "9876543210",
  "OT_PHYSN_NPI": "5555555555",
  "CLM_UTLZTN_DAY_CNT": 10,
  "ADMTNG_ICD9_DGNS_CD": "78650",
  "CLM_DRG_CD": "065",
  "ICD9_PRCDR_CD_1": "3991"
}
```

**Click Execute** → See prediction result:

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

### Step 4b: Test via curl (Command Line)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"PRVDR_NUM\": \"1002CR\", \"NCH_PRMRY_PYR_CLM_PD_AMT\": 25000.0, \"AT_PHYSN_NPI\": \"1234567890\", \"OP_PHYSN_NPI\": \"9876543210\", \"OT_PHYSN_NPI\": \"5555555555\", \"CLM_UTLZTN_DAY_CNT\": 10, \"ADMTNG_ICD9_DGNS_CD\": \"78650\", \"CLM_DRG_CD\": \"065\", \"ICD9_PRCDR_CD_1\": \"3991\"}"
```

Or save to file and use:

```bash
# Create file: claim.json with the JSON above
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @claim.json
```

### Step 4c: Batch Test (Python)

```bash
# Terminal 3: Test batch endpoint
python -c "
import requests
import json

claims = [
    {
        'PRVDR_NUM': '1002CR',
        'NCH_PRMRY_PYR_CLM_PD_AMT': 25000.0,
        'AT_PHYSN_NPI': '1234567890',
        'OP_PHYSN_NPI': '9876543210',
        'OT_PHYSN_NPI': '5555555555',
        'CLM_UTLZTN_DAY_CNT': 10,
        'ADMTNG_ICD9_DGNS_CD': '78650',
        'CLM_DRG_CD': '065',
        'ICD9_PRCDR_CD_1': '3991'
    },
    {
        'PRVDR_NUM': '5678',
        'NCH_PRMRY_PYR_CLM_PD_AMT': 95000.0,
        'AT_PHYSN_NPI': '1111111111',
        'OP_PHYSN_NPI': '2222222222',
        'OT_PHYSN_NPI': 'UNKNOWN',
        'CLM_UTLZTN_DAY_CNT': 45,
        'ADMTNG_ICD9_DGNS_CD': '78999',
        'CLM_DRG_CD': '999',
        'ICD9_PRCDR_CD_1': '9999'
    }
]

response = requests.post('http://localhost:8000/predict_batch', json=claims)
print(json.dumps(response.json(), indent=2))
"
```

**Output shows:**
- Individual predictions
- Summary: 2 total, 1 anomaly, 1 alert

---

## Phase 5: Demo Business Logic

### Step 5: Run Example Workflow

```bash
python scripts/example_workflow.py
```

**What it shows:**
- How predictions flow through business logic
- Risk scores (0-100%)
- Alert creation logic
- Provider statistics
- Explanations
- Routing decisions (AUTO/REVIEW/ALERT)

---

## Phase 6: Send to Your Web Partner

### Send Them This Info:

```markdown
# API for Your Web Application

## How to Call It

Your frontend should make HTTP requests to:
- **URL:** http://your-server:8000/predict
- **Method:** POST
- **Content-Type:** application/json

## Example Request (JavaScript)

```javascript
const claim = {
  "PRVDR_NUM": "1002CR",
  "NCH_PRMRY_PYR_CLM_PD_AMT": 25000.0,
  "AT_PHYSN_NPI": "1234567890",
  "OP_PHYSN_NPI": "9876543210",
  "OT_PHYSN_NPI": "UNKNOWN",
  "CLM_UTLZTN_DAY_CNT": 10,
  "ADMTNG_ICD9_DGNS_CD": "78650",
  "CLM_DRG_CD": "065",
  "ICD9_PRCDR_CD_1": "3991"
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(claim)
})
.then(r => r.json())
.then(result => {
  console.log(`Risk: ${result.risk_percentage}%`);
  if (result.should_alert) {
    console.log('CREATE ALERT');
  }
});
```

## Response Format

```json
{
  "anomaly_score": -0.1234,
  "risk_percentage": 43.83,
  "prediction": "NORMAL",
  "should_alert": false,
  "provider_id": "1002CR",
  "timestamp": "2026-05-23T10:30:45"
}
```

## Database Schema They Need

```sql
-- Claims they submit
CREATE TABLE claims (
  id INT PRIMARY KEY,
  provider_id VARCHAR(50),
  claim_amount FLOAT,
  patient_id INT,
  created_at TIMESTAMP
);

-- Predictions you return
CREATE TABLE predictions (
  id INT PRIMARY KEY,
  claim_id INT,
  anomaly_score FLOAT,
  risk_percentage FLOAT,
  prediction VARCHAR(50),
  should_alert BOOL,
  created_at TIMESTAMP
);

-- Alerts they manage
CREATE TABLE alerts (
  id INT PRIMARY KEY,
  claim_id INT,
  risk_score FLOAT,
  status VARCHAR(50),  -- OPEN, CLOSED, APPROVED, DENIED
  created_at TIMESTAMP,
  resolved_at TIMESTAMP
);
```

## To Deploy

1. Host this API on a cloud server (AWS, Azure, GCP)
2. Change `localhost:8000` to your cloud URL
3. Partner's web app calls your API URL

---
```

---

## Complete Terminal Setup

### Terminal 1: Training (Run once)
```bash
python scripts/run_pipeline.py --input data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --target anomaly_label
```

### Terminal 2: MLflow Dashboard
```bash
mlflow ui
```
→ Open: http://localhost:5000

### Terminal 3: API Server
```bash
python -m uvicorn src.app.api:app --reload --port 8000
```
→ Open: http://localhost:8000/docs

### Terminal 4: Testing
```bash
# Option A: Test via Swagger UI (easiest)
# Go to http://localhost:8000/docs and click "Try it out"

# Option B: Run demo
python scripts/example_workflow.py

# Option C: Manual test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{...your claim data...}'
```

---

## What Goes Where

| Component | Who Builds | Runs Where | Purpose |
|-----------|-----------|-----------|---------|
| **run_pipeline.py** | You | Your computer (once) | Train model |
| **MLflow** | Built-in | Your computer | View metrics |
| **api.py** | You | Your server | Serve predictions |
| **inference.py** | You | Your API | Transform + predict |
| **example_workflow.py** | You | Demo only | Show how it works |
| **Database** | Partner | Their server | Store claims + results |
| **Frontend** | Partner | Their server | User interface |

---

## Troubleshooting

### Problem: "Model not found"
```
Solution: Run run_pipeline.py first to train model
```

### Problem: "Port 8000 already in use"
```bash
# Use different port
python -m uvicorn src.app.api:app --port 8001

# Or kill process using port 8000
# Windows: netstat -ano | findstr :8000
# Then: taskkill /PID <PID> /F
```

### Problem: "Module not found"
```bash
# Make sure you're in the project root:
cd e:\SCHOOL\Health\ Insurance\ Fraud\ Detection

# Then run commands
```

### Problem: MLflow path error
```
Solution: Already fixed in run_pipeline.py (uses direct path now)
```

---

## Success Checklist

- [ ] Pipeline runs successfully → Model trained ✓
- [ ] MLflow dashboard shows metrics ✓
- [ ] API server starts without errors ✓
- [ ] /predict endpoint responds with risk scores ✓
- [ ] /predict_batch handles multiple claims ✓
- [ ] example_workflow.py runs and shows business logic ✓
- [ ] Partner gets API documentation ✓
- [ ] Partner sets up database ✓
- [ ] Partner builds web form → calls your API ✓
- [ ] End-to-end: Claim → Prediction → Alert ✓

---

## Next: Deploy to Cloud

Once everything works locally, deploy to cloud:

```bash
# Example: AWS EC2
# 1. Create EC2 instance (Ubuntu 22.04)
# 2. Clone your repo
# 3. Install Python 3.9+
# 4. Run: python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000

# Partner accesses via: http://your-ec2-ip:8000
```

---

## Summary

```
YOU                              PARTNER
┌─────────────────┐             ┌─────────────────────┐
│ 1. Train model  │ ────────→   │ 2. Build database   │
│    (1 time)     │             │    (once)           │
├─────────────────┤             ├─────────────────────┤
│ 3. Start API    │ ────────→   │ 4. Call your API    │
│    (always on)  │             │    (for each claim) │
├─────────────────┤             ├─────────────────────┤
│ 4. Return JSON  │ ────────→   │ 5. Store result +   │
│                 │             │    Show to user     │
└─────────────────┘             └─────────────────────┘
```

---
