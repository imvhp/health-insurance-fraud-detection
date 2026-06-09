"""
EXAMPLE USAGE - Complete Workflow Integration
==============================================

This script demonstrates how your web partner will call all the business logic modules.

WORKFLOW:
1. Get claims data for a provider
2. Get anomaly scores from ML model
3. Calculate provider risk score
4. Compile provider statistics
5. Generate SHAP explanations
6. Determine if alert should be created
7. Determine routing (AUTO/REVIEW/ALERT)

This is a TEMPLATE for your partner's integration code.
They will adapt this to their web framework (Flask, FastAPI, Django, etc.)

USAGE:
    python scripts/example_workflow.py
    
    This demonstrates the complete flow and shows expected outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import time

# Import business logic modules
from src.models.risk_scoring import (
    calculate_provider_risk_score,
    categorize_risk,
    get_routing_action,
    get_risk_statistics
)
from src.models.provider_service import (
    compile_provider_stats,
    get_provider_risk_context
)
from src.models.explainability import (
    generate_claim_explanation,
    summarize_provider_explanations
)
from src.models.alert_logic import (
    should_create_alert,
    get_alert_priority,
    get_routing_decision
)


def example_workflow():
    """
    Complete end-to-end example of the fraud detection workflow.
    
    This shows how all modules work together.
    """
    
    print("="*70)
    print("FRAUD DETECTION SYSTEM - Complete Workflow Example")
    print("="*70)
    
    # ========== STEP 0: SETUP DATA ==========
    # In reality, your partner gets this from database/API
    
    print("\n[STEP 0] Loading sample claims data...")
    claims_df, anomaly_scores = load_sample_data()
    print(f"Loaded {len(claims_df)} claims")
    
    # Focus on one provider for detailed analysis
    provider_id = "1002CR"
    
    # ========== STEP 1: GET PROVIDER CLAIMS ==========
    print(f"\n[STEP 1] Processing provider: {provider_id}")
    
    provider_claims = claims_df[claims_df['PRVDR_NUM'] == provider_id]
    provider_anomaly_scores = anomaly_scores[claims_df['PRVDR_NUM'] == provider_id]
    
    print(f"Provider has {len(provider_claims)} claims")
    
    # ========== STEP 2: CALCULATE PROVIDER RISK SCORE ==========
    print(f"\n[STEP 2] Calculating provider risk score...")
    
    # Convert model outputs (-1/1) to 0-100% scale
    # Example: Convert Isolation Forest output to percentile
    anomaly_percentiles = np.clip((provider_anomaly_scores + 1) / 2 * 100, 0, 100)
    
    print(f"Individual claim anomaly scores (0-100%): {anomaly_percentiles[:5]}...")
    
    # Calculate composite provider risk
    provider_risk_score, risk_category = calculate_provider_risk_score(anomaly_percentiles)
    
    print(f"✓ Provider Risk Score: {provider_risk_score:.1f}%")
    print(f"✓ Risk Category: {risk_category}")
    
    # Get routing action
    routing_action = get_routing_action(provider_risk_score)
    print(f"✓ Routing Action: {routing_action}")
    
    # ========== STEP 3: COMPILE PROVIDER STATISTICS ==========
    print(f"\n[STEP 3] Compiling provider statistics...")
    
    provider_stats = compile_provider_stats(provider_id, claims_df)
    
    print(f"Total Claims: {provider_stats['total_claims']}")
    print(f"Total Cost: ${provider_stats['total_cost']:,.2f}")
    print(f"Unique Patients: {provider_stats['unique_patients']}")
    print(f"Average Claim: ${provider_stats['average_claim_amount']:,.2f}")
    print(f"Anomaly Count: {provider_stats['anomaly_count']}")
    print(f"Anomaly Rate: {provider_stats['anomaly_rate']}%")
    
    # ========== STEP 4: GENERATE EXPLANATIONS ==========
    print(f"\n[STEP 4] Generating explanations...")
    
    # Explain why first flagged claim is anomalous
    flagged_indices = np.where(anomaly_percentiles >= 50)[0]
    
    if len(flagged_indices) > 0:
        first_flagged_idx = flagged_indices[0]
        flagged_claim = provider_claims.iloc[first_flagged_idx].to_dict()
        
        print(f"Explaining claim #{first_flagged_idx}...")
        # Note: In real use, would pass actual trained model here
        # For this example, we'll create dummy explanation
        explanation = {
            "top_factors": [
                {"feature": "claim_amount", "impact": 0.45, "direction": "high", "value": flagged_claim.get('CLM_PMT_AMT', 'N/A')},
                {"feature": "drg_code", "impact": 0.35, "direction": "unusual", "value": flagged_claim.get('CLM_DRG_CD', 'N/A')},
            ],
            "summary": f"Claim {first_flagged_idx} flagged: Unusually high amount (0.45 impact) + rare DRG code (0.35 impact)",
            "confidence": 0.92
        }
        
        print(f"✓ Top Factors:")
        for factor in explanation['top_factors']:
            print(f"  - {factor['feature']}: {factor['direction']} (impact: {factor['impact']}, value: {factor['value']})")
        print(f"✓ Summary: {explanation['summary']}")
    
    # Get provider-level explanation summary
    provider_summary = summarize_provider_explanations(
        provider_id,
        claims_df,
        anomaly_percentiles
    )
    
    print(f"\n✓ Provider Risk Pattern Summary:")
    print(provider_summary['summary'])
    
    # ========== STEP 5: DETERMINE ALERT ACTION ==========
    print(f"\n[STEP 5] Determining alert action...")
    
    should_alert = should_create_alert(provider_risk_score)
    print(f"Should create alert? {should_alert}")
    
    if should_alert:
        priority = get_alert_priority(
            provider_risk_score,
            flagged_claim_count=provider_stats['anomaly_count']
        )
        print(f"Alert Priority: {priority}")
        
        routing = get_routing_decision(provider_risk_score, risk_category)
        print(f"Routing Decision: {routing['action']}")
        print(f"  Queue: {routing['queue']}")
        print(f"  SLA: {routing['sla_hours']} hours")
        print(f"  Reason: {routing['reasoning']}")
    
    # ========== STEP 6: SHOW WHAT WEB PARTNER DOES NEXT ==========
    print(f"\n[STEP 6] What your web partner does next...")
    
    if should_alert:
        print(f"""
YOUR PARTNER SHOULD:
1. Create Alert record in their database:
   - alert_id = f"ALERT_{provider_id}_{int(time.time()*1000)}"
   - provider_id: {provider_id}
   - risk_score: {provider_risk_score:.1f}%
   - status: OPEN
   - priority: {priority}

2. Assign to investigator queue (status = "OPEN")

3. Create notification for investigator

4. When investigator reviews, they see:
   - Provider stats: {provider_stats}
   - Risk factors: {provider_summary['summary']}
   - Claim explanations (like the one above)

5. Investigator makes decision:
   - APPROVED: Claims are fine
   - DENIED: Flag for fraud team
   - NEEDS_MORE_INFO: Request more data

6. Your partner logs decision in DecisionLog
        """)
    else:
        print(f"Provider is LOW-RISK, proceed with auto-approval")
    
    print("\n" + "="*70)


def load_sample_data():
    """
    Create sample claims data for demonstration.
    
    In real use, this comes from database.
    """
    
    np.random.seed(42)
    
    # Create sample claims
    n_claims = 100
    provider_ids = ["1002CR", "5678", "9012"] * (n_claims // 3)
    
    claims_data = {
        'CLM_ID': range(n_claims),
        'PRVDR_NUM': provider_ids[:n_claims],
        'DESYNPUF_ID': np.random.randint(1, 50, n_claims),
        'CLM_PMT_AMT': np.random.exponential(5000, n_claims),
        'NCH_PRMRY_PYR_CLM_PD_AMT': np.random.exponential(5000, n_claims),
        'CLM_UTLZTN_DAY_CNT': np.random.poisson(5, n_claims),
        'CLM_DRG_CD': np.random.randint(100, 500, n_claims),
        'ADMTNG_ICD9_DGNS_CD': np.random.randint(1000, 10000, n_claims),
        'is_anomaly': np.random.choice(['NORMAL', 'ANOMALY'], n_claims, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(claims_data)
    
    # Generate corresponding anomaly scores (-1 to 1, like Isolation Forest)
    anomaly_scores = np.random.uniform(-1, 1, n_claims)
    # Make ANOMALY records actually anomalous
    anomaly_mask = df['is_anomaly'] == 'ANOMALY'
    anomaly_scores[anomaly_mask] = np.random.uniform(0.5, 1, anomaly_mask.sum())
    
    return df, anomaly_scores


def example_single_claim_explanation():
    """
    Simpler example: just explain a single claim.
    """
    
    print("\n" + "="*70)
    print("SIMPLE EXAMPLE: Explaining a Single Claim")
    print("="*70)
    
    # Single claim to explain
    claim_features = {
        "claim_amount": 25000,
        "hospital_days": 10,
        "drg_code": 999,  # Rare code
        "diagnosis_code": 78650
    }
    
    print(f"\nClaim to explain: {claim_features}")
    
    # In real use, would pass trained model
    # For now, just show structure
    explanation = {
        "top_factors": [
            {"feature": "claim_amount", "impact": 0.45, "direction": "high", "value": 25000},
            {"feature": "drg_code", "impact": 0.35, "direction": "unusual", "value": 999},
        ],
        "summary": "Flagged because claim amount is 5x average (0.45 impact) + DRG code is extremely rare (0.35 impact)",
        "confidence": 0.92,
        "method": "SHAP"
    }
    
    print(f"\nExplanation:")
    print(f"  Summary: {explanation['summary']}")
    print(f"  Confidence: {explanation['confidence']}")
    print(f"  Method: {explanation['method']}")
    print(f"\n  Top Contributing Factors:")
    for factor in explanation['top_factors']:
        print(f"    • {factor['feature']}: {factor['direction']} (impact: {factor['impact']})")


if __name__ == "__main__":
    # Run complete workflow example
    example_workflow()
    
    # Run simple single-claim example
    example_single_claim_explanation()
