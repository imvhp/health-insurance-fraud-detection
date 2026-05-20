# 🛡️ Health Insurance Fraud Detection

A machine learning project that detects fraudulent Medicare inpatient claims using **Isolation Forest** algorithm. The system analyzes claim characteristics to identify structural and attributive anomalies.

---

## 📋 Project Overview

This project implements an end-to-end ML pipeline to identify potentially fraudulent Medicare claims. It uses unsupervised anomaly detection to flag suspicious patterns without requiring labeled fraud data.

**Dataset**: Medicare inpatient claims data (2008-2010)  
**Model**: Isolation Forest (Scikit-learn)  
**Deployment**: FastAPI backend + Streamlit frontend  
**Tracking**: MLflow for experiment management

---

## 🔄 Project Flow (Simplified)

```
Raw Data
   ↓
Load Data (CSV)
   ↓
Preprocess (cleaning, filling missing values)
   ↓
Validate Data (Great Expectations)
   ↓
Feature Engineering (select & encode features)
   ↓
Train Model (Isolation Forest)
   ↓
Serve Predictions (FastAPI API)
   ↓
Display Results (Streamlit UI)
```

### Detailed Pipeline Stages:

1. **Data Loading** → Reads Medicare claims CSV
2. **Preprocessing** → Cleans data, handles missing values, injects test anomalies
3. **Data Validation** → Ensures data quality and schema compliance
4. **Feature Selection** → Selects 9 baseline features (provider, physician, financials, diagnosis codes)
5. **Feature Encoding** → Converts categorical data to numerical format
6. **Model Training** → Trains Isolation Forest with MLflow tracking
7. **Prediction** → Scores claims and identifies anomalies
8. **API Serving** → FastAPI backend exposes `/predict` endpoint
9. **Web UI** → Streamlit interface for interactive predictions

---

## 📁 Project Structure

```
├── src/                          # Core source code
│   ├── data/
│   │   ├── load_data.py         # Data loading utilities
│   │   └── preprocess.py        # Data preprocessing, anomaly injection
│   ├── features/
│   │   └── build_features.py    # Feature selection & encoding
│   ├── models/
│   │   ├── train.py             # Model training
│   │   ├── tune.py              # Hyperparameter tuning
│   │   └── evaluate.py          # Model evaluation metrics
│   ├── serving/
│   │   └── inference.py         # Prediction logic
│   ├── app/
│   │   ├── api.py               # FastAPI backend
│   │   └── app.py               # Streamlit frontend
│   └── utils/
│       └── validate_data.py     # Data validation rules
├── scripts/
│   ├── run_pipeline.py          # Main training pipeline (orchestrates all stages)
│   ├── test_pipeline_phase1_data_features.py  # Tests data & features
│   └── test_pipeline_phase2_modeling.py       # Tests model training
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
├── data/
│   ├── raw/                     # Original Medicare claims CSV
│   └── processed/               # Cleaned, preprocessed data
├── configs/                     # Configuration files
├── docker/                      # Docker setup files
├── great_expectations/          # Data validation configs
├── mlruns/                      # MLflow experiment tracking
├── artifacts/                   # Saved models & artifacts
└── tests/                       # Unit & integration tests
```

---

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn mlflow streamlit fastapi uvicorn requests
```

### 2. **Run the Training Pipeline**

Trains the Isolation Forest model on Medicare claims data:

```bash
python scripts/run_pipeline.py \
    --experiment "fraud_detection" \
    --test_size 0.2 \
    --anomaly_fraction 0.05
```

**Pipeline does:**
- Loads raw data
- Preprocesses it
- Extracts 9 baseline features
- Encodes categorical variables
- Trains Isolation Forest
- Logs metrics to MLflow

### 3. **Start FastAPI Backend**

```bash
python -m uvicorn src.app.api:app --reload --port 8000
```

API endpoint: `http://localhost:8000/predict`

### 4. **Launch Streamlit Frontend**

In another terminal:

```bash
streamlit run src/app/app.py
```

UI: `http://localhost:8501`

---

## 📊 Model Details

### Algorithm: Isolation Forest

- **Type**: Unsupervised anomaly detection
- **Why**: Works well for high-dimensional data, detects global & local outliers
- **Input Features** (9 baseline):
  - `PRVDR_NUM` - Provider identifier
  - `AT_PHYSN_NPI` - Attending physician ID
  - `OP_PHYSN_NPI` - Operating physician ID
  - `OT_PHYSN_NPI` - Other physician ID
  - `NCH_PRMRY_PYR_CLM_PD_AMT` - Claim amount paid
  - `CLM_UTLZTN_DAY_CNT` - Hospital utilization days
  - `ADMTNG_ICD9_DGNS_CD` - Admission diagnosis code
  - `CLM_DRG_CD` - DRG code
  - `ICD9_PRCDR_CD_1` - Primary procedure code

### Output

- **Anomaly Score**: Lower score = more anomalous
- **Prediction**: -1 (anomaly/fraud), +1 (normal)
- **Metrics Tracked**: Precision, Recall, F1-Score, ROC-AUC

---

## 🧪 Testing

Run validation tests:

```bash
# Phase 1: Data & Features validation
python scripts/test_pipeline_phase1_data_features.py

# Phase 2: Model training validation
python scripts/test_pipeline_phase2_modeling.py
```

---

## 📈 Experiment Tracking

View all training runs and metrics in MLflow:

```bash
mlflow ui
```

Opens at `http://localhost:5000`

Tracks:
- Hyperparameters
- Performance metrics
- Model artifacts
- Training logs

---

## 🔧 Configuration

Key configuration options in `run_pipeline.py`:

```python
--experiment          # MLflow experiment name
--test_size          # Train/test split ratio (default: 0.2)
--anomaly_fraction   # Expected fraud rate (default: 0.05)
--mlflow_uri         # MLflow tracking URI
```

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| ML Framework | Scikit-learn |
| Model | Isolation Forest |
| Data Processing | Pandas, NumPy |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| Containerization | Docker |

---

## 📝 Key Features

✅ **End-to-end ML pipeline** - From raw data to predictions  
✅ **Interactive web UI** - Streamlit frontend for easy use  
✅ **REST API** - FastAPI backend for integration  
✅ **Experiment tracking** - MLflow for reproducibility  
✅ **Data validation** - Great Expectations quality checks  
✅ **Docker support** - Containerized deployment  
✅ **Automated testing** - Phase-based pipeline tests  

---

## 📞 Usage Example

### Via Web UI (Easiest)
1. Launch Streamlit: `streamlit run src/app/app.py`
2. Fill in claim details
3. Click "Analyze Claim"
4. View fraud risk score

### Via API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PRVDR_NUM": "1002CR",
    "NCH_PRMRY_PYR_CLM_PD_AMT": 15000.0,
    "AT_PHYSN_NPI": "9876543210",
    "OP_PHYSN_NPI": "missing",
    "OT_PHYSN_NPI": "missing",
    "CLM_UTLZTN_DAY_CNT": 5,
    "ADMTNG_ICD9_DGNS_CD": "78650",
    "CLM_DRG_CD": "470",
    "ICD9_PRCDR_CD_1": "8152"
  }'
```

---

## 📚 Project Insights

- **Unsupervised Learning**: No labeled fraud data needed
- **Scalable**: Handles large datasets efficiently  
- **Explainable**: Isolation Forest provides anomaly scores
- **Real-time**: Fast inference for production use
- **Modular**: Easily update features, model, or pipeline stages

---

## 🚀 Future Enhancements

- [ ] Add additional unsupervised models (Isolation Forest ensemble)
- [ ] Implement supervised learning for labeled fraud data
- [ ] Dashboard for monitoring predictions in production
- [ ] Automated retraining pipeline
- [ ] Model explainability (SHAP values)
- [ ] Alert system for high-risk claims

---

## 📄 License

Educational project - Use for learning purposes.

---

**Last Updated**: May 2026  
**Author**: Health Insurance Fraud Detection Team
