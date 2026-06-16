from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates an Isolation Forest model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    raw_preds = model.predict(X_test)
    # Isolation Forest outputs -1 for anomalies, 1 for normal. Map to 1 and 0 to match y_test.
    preds = np.where(raw_preds == -1, 1, 0)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))