#!/usr/bin/env python3
"""
SIMPLE API TEST SCRIPT
======================

Tests all endpoints of the Medicare Fraud Detection API.

USAGE:
  1. Make sure API is running: python -m uvicorn src.app.api:app --reload --port 8000
  2. Run this script: python scripts/test_api.py
"""

import requests
import json
from typing import Dict, Any

# API server URL
API_URL = "http://localhost:8000"

# Test claim data (valid JSON format)
VALID_CLAIM = {
    "PRVDR_NUM": "1151",
    "NCH_PRMRY_PYR_CLM_PD_AMT": 0.0,
    "AT_PHYSN_NPI": "16227",
    "OP_PHYSN_NPI": "7750",
    "OT_PHYSN_NPI": "4386",
    "CLM_UTLZTN_DAY_CNT": 10.0,
    "ADMTNG_ICD9_DGNS_CD": "2261",
    "CLM_DRG_CD": "705",
    "ICD9_PRCDR_CD_1": "1152"
}

HIGH_RISK_CLAIM = {
    "PRVDR_NUM": "1632",
    "NCH_PRMRY_PYR_CLM_PD_AMT": 34592.88887814058,  # Unusually high!
    "AT_PHYSN_NPI": "10378",
    "OP_PHYSN_NPI": "7333",
    "OT_PHYSN_NPI": "4386",
    "CLM_UTLZTN_DAY_CNT": 5.0,  # Very long stay
    "ADMTNG_ICD9_DGNS_CD": "263",
    "CLM_DRG_CD": "76",  # Rare code
    "ICD9_PRCDR_CD_1": "15"
}

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_result(response_text: str, response_json: Dict[Any, Any], status_code: int):
    """Print formatted result"""
    print(f"\n✓ Status Code: {status_code}")
    print(f"✓ Response:")
    print(json.dumps(response_json, indent=2))

def test_root():
    """Test GET /"""
    print_section("TEST 1: Health Check (GET /)")
    try:
        response = requests.get(f"{API_URL}/")
        print_result("GET /", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test GET /health"""
    print_section("TEST 2: Health Endpoint (GET /health)")
    try:
        response = requests.get(f"{API_URL}/health")
        print_result("GET /health", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_status():
    """Test GET /model/status"""
    print_section("TEST 3: Model Status (GET /model/status)")
    try:
        response = requests.get(f"{API_URL}/model/status")
        print_result("GET /model/status", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_single():
    """Test POST /predict with single claim"""
    print_section("TEST 4: Single Prediction (POST /predict)")
    print(f"\nClaim Data (valid JSON):")
    print(json.dumps(VALID_CLAIM, indent=2))
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=VALID_CLAIM,
            headers={"Content-Type": "application/json"}
        )
        print_result("POST /predict", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return False

def test_predict_high_risk():
    """Test POST /predict with high-risk claim"""
    print_section("TEST 5: High-Risk Claim (POST /predict)")
    print(f"\nClaim Data (high risk):")
    print(json.dumps(HIGH_RISK_CLAIM, indent=2))
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=HIGH_RISK_CLAIM,
            headers={"Content-Type": "application/json"}
        )
        print_result("POST /predict", response.json(), response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("should_alert"):
                print("\n⚠️  ALERT TRIGGERED for high-risk claim!")
                return True
            else:
                print("\n✓ Prediction received (no alert for this claim)")
                return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return False

def test_predict_batch():
    """Test POST /predict_batch with multiple claims"""
    print_section("TEST 6: Batch Prediction (POST /predict_batch)")
    
    batch_claims = [VALID_CLAIM, HIGH_RISK_CLAIM]
    print(f"\nBatch Size: 2 claims")
    print(f"Claim 1: Normal risk")
    print(f"Claim 2: High risk")
    
    try:
        response = requests.post(
            f"{API_URL}/predict_batch",
            json=batch_claims,
            headers={"Content-Type": "application/json"}
        )
        print_result("POST /predict_batch", response.json(), response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Batch Stats:")
            print(f"  - Total Claims: {result.get('total')}")
            print(f"  - Anomalies Detected: {result.get('anomalies_detected')}")
            print(f"  - Alerts Triggered: {result.get('alerts_triggered')}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return False

def test_explain():
    """Test POST /explain"""
    print_section("TEST 7: Explain Prediction (POST /explain)")
    print(f"\nClaim Data:")
    print(json.dumps(VALID_CLAIM, indent=2))
    
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json=VALID_CLAIM,
            headers={"Content-Type": "application/json"}
        )
        print_result("POST /explain", response.json(), response.status_code)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" MEDICARE FRAUD DETECTION API - TEST SUITE")
    print("="*70)
    print(f"\nTesting API at: {API_URL}")
    print("Make sure API is running:")
    print("  python -m uvicorn src.app.api:app --reload --port 8000")
    
    # Run all tests
    results = {
        "Health Check": test_root(),
        "Health Endpoint": test_health(),
        "Model Status": test_model_status(),
        "Single Prediction": test_predict_single(),
        "High-Risk Prediction": test_predict_high_risk(),
        "Batch Prediction": test_predict_batch(),
        "Explain Prediction": test_explain(),
    }
    
    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! API is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check API and try again.")

if __name__ == "__main__":
    main()
