"""Assertion tests for score range correctness."""
import sys, os
# Run from project root
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from src.serving.inference import predict_claims

def check(pred, label=""):
    score = pred["anomaly_score"]
    risk = pred["risk_percentage"]
    assert -1.0 <= score <= 1.0, f"{label} Score out of [-1,1] range: {score}"
    assert 0.0 <= risk <= 100.0, f"{label} Risk out of [0,100] range: {risk}"
    # Consistency: negative score (anomaly) should give risk > 50
    if score < 0:
        assert risk > 50.0, f"{label} Negative score but risk <= 50: score={score} risk={risk}"
    if score > 0:
        assert risk < 50.0, f"{label} Positive score but risk >= 50: score={score} risk={risk}"
    return True

# Test 1: low-risk claim (low payment, short stay)
test1 = [{"PRVDR_NUM": "330383", "CLM_PMT_AMT": 500.0, "CLM_UTLZTN_DAY_CNT": 1}]
res1 = predict_claims(test1)
pred1 = res1["predictions"][0]
print(f"[TEST1-LOW]  model={res1['model_selected']} score={pred1['anomaly_score']:.4f} risk={pred1['risk_percentage']}% pred={pred1['prediction']}")
check(pred1, "TEST1")

# Test 2: high-risk claim
test2 = [{"PRVDR_NUM": "330383", "CLM_PMT_AMT": 500000.0, "CLM_UTLZTN_DAY_CNT": 60}]
res2 = predict_claims(test2)
pred2 = res2["predictions"][0]
print(f"[TEST2-HIGH] model={res2['model_selected']} score={pred2['anomaly_score']:.4f} risk={pred2['risk_percentage']}% pred={pred2['prediction']}")
check(pred2, "TEST2")

# Test 3: batch
res3 = predict_claims(test1 + test2)
print(f"[TEST3-BATCH] model={res3['model_selected']} count={len(res3['predictions'])}")
for i, p in enumerate(res3["predictions"]):
    print(f"  claim {i}: score={p['anomaly_score']:.4f} risk={p['risk_percentage']}%")
    check(p, f"TEST3-claim{i}")

# Test 4: risk_scoring module
from src.risk_scoring import calculate_provider_risk_score, categorize_risk, get_routing_action, get_risk_statistics  # noqa
scores_in = [pred1["risk_percentage"], pred2["risk_percentage"]]
prov_score, cat = calculate_provider_risk_score(scores_in)
print(f"[TEST4-RISK_SCORING] provider_score={prov_score:.2f} category={cat}")
assert 0 <= prov_score <= 100
stats = get_risk_statistics(scores_in)
print(f"  stats={stats}")

# Test 5: alert_logic module
from src.alert_logic import should_create_alert, get_alert_priority, get_routing_decision  # noqa
alert = should_create_alert(prov_score)
print(f"[TEST5-ALERT] should_alert={alert}")
priority = get_alert_priority(prov_score)
routing = get_routing_decision(prov_score, cat)
print(f"  priority={priority} routing_action={routing['action']} sla={routing.get('sla_hours')}h")

print("\n[ALL ASSERTIONS PASSED]")
