#!/usr/bin/env python3
"""
API Test Script for PTL FastAPI server
======================================

USAGE:
  1. Make sure PTL API is running: python -m uvicorn src.app.api:app --reload --port 8002
  2. Run this script: python scripts/test_api_ptl.py
"""

import requests
import json
import sys
import io

# Fix console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

API_URL = "http://localhost:8002"

VALID_CLAIM = {
    "DESYNPUF_ID": "000138C7A00C4174",
    "CLM_ID": "996601974052374",
    "SEGMENT": 1,
    "CLM_FROM_DT": 20080312,
    "CLM_THRU_DT": 20080323,
    "PRVDR_NUM": "330383",
    "CLM_PMT_AMT": 26000.0,
    "NCH_PRMRY_PYR_CLM_PD_AMT": 0.0,
    "AT_PHYSN_NPI": "321323328.0",
    "CLM_ADMSN_DT": 20080312,
    "ADMTNG_ICD9_DGNS_CD": "4580",
    "CLM_PASS_THRU_PER_DIEM_AMT": 0.0,
    "NCH_BENE_IP_DDCTBL_AMT": 1024.0,
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM": 0.0,
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM": 0.0,
    "CLM_UTLZTN_DAY_CNT": 11,
    "NCH_BENE_DSCHRG_DT": 20080323,
    "CLM_DRG_CD": "314",
    "ICD9_DGNS_CD_1": "78559"
}

HIGH_RISK_CLAIM = {
    "DESYNPUF_ID": "000138C7A00C4174",
    "CLM_ID": "996601974052375",
    "SEGMENT": 1,
    "CLM_FROM_DT": 20080312,
    "CLM_THRU_DT": 20080328,
    "PRVDR_NUM": "330383",
    "CLM_PMT_AMT": 250000.0, # Unusually high payment!
    "NCH_PRMRY_PYR_CLM_PD_AMT": 150000.0,
    "AT_PHYSN_NPI": "321323328.0",
    "CLM_ADMSN_DT": 20080312,
    "ADMTNG_ICD9_DGNS_CD": "9999", # Rare/unseen code
    "CLM_PASS_THRU_PER_DIEM_AMT": 0.0,
    "NCH_BENE_IP_DDCTBL_AMT": 1024.0,
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM": 0.0,
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM": 0.0,
    "CLM_UTLZTN_DAY_CNT": 45, # Extremely long stay
    "NCH_BENE_DSCHRG_DT": 20080328,
    "CLM_DRG_CD": "999", # Rare DRG
    "ICD9_DGNS_CD_1": "99999"
}

def print_section(title: str):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_result(response_text: str, response_json: dict, status_code: int):
    print(f"\n[OK] Status Code: {status_code}")
    print(f"[OK] Response:")
    print(json.dumps(response_json, indent=2))

def test_root():
    print_section("TEST 1: Health Check (GET /)")
    try:
        response = requests.get(f"{API_URL}/")
        print_result("GET /", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_health():
    print_section("TEST 2: Health Endpoint (GET /health)")
    try:
        response = requests.get(f"{API_URL}/health")
        print_result("GET /health", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_model_status():
    print_section("TEST 3: Model Status (GET /model/status)")
    try:
        response = requests.get(f"{API_URL}/model/status")
        print_result("GET /model/status", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_predict_single():
    print_section("TEST 4: Single Prediction (POST /predict)")
    try:
        response = requests.post(f"{API_URL}/predict", json=VALID_CLAIM)
        data = response.json()
        print_result("POST /predict", data, response.status_code)
        assert response.status_code == 200
        assert "model_selected" in data, "Missing model_selected field"
        assert 0 <= data["risk_percentage"] <= 100, f"risk_percentage out of range: {data['risk_percentage']}"
        print(f"  [ASSERT] model={data['model_selected']}  risk={data['risk_percentage']}%")
        return True
    except AssertionError as e:
        print(f"[FAIL] Assertion: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_predict_high_risk():
    print_section("TEST 5: High-Risk Claim (POST /predict)")
    try:
        response = requests.post(f"{API_URL}/predict", json=HIGH_RISK_CLAIM)
        data = response.json()
        print_result("POST /predict", data, response.status_code)
        assert response.status_code == 200
        assert "model_selected" in data, "Missing model_selected field"
        print(f"  [ASSERT] model={data['model_selected']}  risk={data['risk_percentage']}%")
        return True
    except AssertionError as e:
        print(f"[FAIL] Assertion: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_predict_batch():
    print_section("TEST 6: Batch Prediction (POST /predict_batch)")
    batch_claims = [VALID_CLAIM, HIGH_RISK_CLAIM]
    try:
        response = requests.post(f"{API_URL}/predict_batch", json=batch_claims)
        data = response.json()
        print_result("POST /predict_batch", data, response.status_code)
        assert response.status_code == 200
        for i, pred in enumerate(data.get("predictions", [])):
            assert "model_selected" in pred, f"Claim {i} missing model_selected"
            assert 0 <= pred["risk_percentage"] <= 100
            print(f"  [ASSERT] claim {i}: model={pred['model_selected']}  risk={pred['risk_percentage']}%")
        return True
    except AssertionError as e:
        print(f"[FAIL] Assertion: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_explain():
    print_section("TEST 7: Explain Prediction (POST /explain) [SHAP]")
    try:
        response = requests.post(f"{API_URL}/explain", json=HIGH_RISK_CLAIM)
        data = response.json()
        print_result("POST /explain", data, response.status_code)
        assert response.status_code == 200

        # ── Top-level fields ──────────────────────────────────────────────────
        assert "top_factors" in data,  "Missing top_factors in explain response"
        assert "summary"     in data,  "Missing summary in explain response"
        assert "confidence"  in data,  "Missing confidence in explain response"
        assert "method"      in data,  "Missing method in explain response"
        assert "provider_id" in data,  "Missing provider_id in explain response"

        # ── method must be "SHAP" or "Heuristic" ─────────────────────────────
        assert data["method"] in ("SHAP", "Heuristic"), \
            f"Unexpected explain method: {data['method']}"

        # ── confidence must be in [0, 1] ──────────────────────────────────────
        assert 0.0 <= data["confidence"] <= 1.0, \
            f"confidence out of range: {data['confidence']}"

        # ── top_factors schema validation ─────────────────────────────────────
        assert isinstance(data["top_factors"], list), "top_factors should be a list"
        for i, factor in enumerate(data["top_factors"]):
            assert "feature"   in factor, f"top_factors[{i}] missing 'feature'"
            assert "impact"    in factor, f"top_factors[{i}] missing 'impact'"
            assert "direction" in factor, f"top_factors[{i}] missing 'direction'"
            assert "value"     in factor, f"top_factors[{i}] missing 'value'"
            assert factor["direction"] in (
                "increases anomaly risk", "decreases anomaly risk"
            ), f"top_factors[{i}] unexpected direction: {factor['direction']}"
            assert factor["impact"] >= 0, \
                f"top_factors[{i}] impact should be non-negative: {factor['impact']}"

        # ── summary should be a non-empty string ─────────────────────────────
        assert isinstance(data["summary"], str) and len(data["summary"]) > 0, \
            "summary should be a non-empty string"

        # ── print per-factor breakdown for easy debugging ─────────────────────
        print(f"\n  [ASSERT] method={data['method']}  confidence={data['confidence']}")
        print(f"  [ASSERT] provider_id={data['provider_id']}")
        for i, f in enumerate(data["top_factors"]):
            print(f"  [FACTOR {i+1}] {f['feature']}: impact={f['impact']}  "
                  f"dir={f['direction']}  val={f['value']}")

        return True
    except AssertionError as e:
        print(f"[FAIL] Assertion: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def main():
    print("\n" + "="*70)
    print(" CMS FRAUD DETECTION API (PTL) - TEST SUITE")
    print("="*70)
    
    results = {
        "Health Check": test_root(),
        "Health Endpoint": test_health(),
        "Model Status": test_model_status(),
        "Single Prediction": test_predict_single(),
        "High-Risk Prediction": test_predict_high_risk(),
        "Batch Prediction": test_predict_batch(),
        "Explain Prediction": test_explain(),
    }
    
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"{status}: {test_name}")
        
    print(f"\nTotal: {passed}/{total} tests passed")
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! PTL API is working correctly.")
        sys.exit(0)
    else:
        print(f"\n[WARNING] Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
