import optuna
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def tune_model(X, y):
    """
    Tunes an Isolation Forest model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target (anomaly labels).
    """
    # Split the dataset to validate hyperparameters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "contamination": trial.suggest_float("contamination", 0.01, 0.05),
            "random_state": 42,
            "n_jobs": -1
        }
        
        model = IsolationForest(**params)
        model.fit(X_train)
        
        # Invert decision function output so higher = more anomalous (ROC-AUC standard)
        anomaly_scores = -1 * model.decision_function(X_test)
        
        auc = roc_auc_score(y_test, anomaly_scores)
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params