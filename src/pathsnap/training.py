
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

FEATURES = [
    "candidate_rank",
    "distance_m",
    "gps_heading_deg",
    "arc_heading_deg",
    "heading_error_deg"
]


def load_training_data():
    candidates = pd.read_csv(
        "data/candidate_handoff/train_candidates.csv.gz",
        dtype={"trip_id": "string"}
    )

    labels = pd.read_csv(
        "results/aligned_labels_train.csv",
        dtype={"trip_id": "string"}
    )

    data = candidates.merge(
        labels,
        on=["split", "trip_id", "observation_index"],
        how="left",
        validate="many_to_one"
    )

    if data["true_arc_id"].isna().any():
        raise ValueError("Some candidate rows have no aligned label.")

    data["target"] = (
        data["arc_id"] == data["true_arc_id"]
    ).astype("int8")

    return data


def build_models():
    dt_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    hgb_model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=42
    )

    return {
        "Decision Tree": dt_model,
        "Histogram Gradient Boosting": hgb_model
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions,
            zero_division=0
        )
    }


def train_and_evaluate():
    data = load_training_data()

    X = data[FEATURES]
    y = data["target"]
    groups = data["trip_id"]

    # Keep entire trips together during validation.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        results[name] = evaluate_model(
            model, X_test, y_test
        )

        joblib.dump(
            model,
            f"results/{name.lower().replace(' ', '_')}.joblib"
        )

        print(f"{name} results:")
        print(results[name])

    comparison = pd.DataFrame.from_dict(
        results, orient="index"
    )

    comparison.to_csv(
        "results/krithika_model_comparison.csv"
    )

    print("\nMODEL COMPARISON")
    print(comparison)

    print("\nValidation trips:")
    print(groups.iloc[test_idx].nunique())

    print("\nTraining completed. Models saved in results/")

    return results


if __name__ == "__main__":
    train_and_evaluate()
