import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
def train_models(X_train, y_train):
    # Decision Tree model
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )

    dt_model.fit(X_train, y_train)

    # Histogram Gradient Boosting model
    hgb_model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    )

    hgb_model.fit(X_train, y_train)

    return dt_model, hgb_model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    }

    return results
def compare_models(dt_results, hgb_results):
    comparison = pd.DataFrame([
        {"Model": "Decision Tree", **dt_results},
        {"Model": "Histogram Gradient Boosting", **hgb_results}
    ])

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    return comparison