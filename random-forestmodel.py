import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
#   Configuration & Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Output directories
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 30


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data — in real projects this would read from CSV/Parquet/database"""
    logger.info("Loading Iris dataset (placeholder for real data loading)")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    # In real life you would do:
    # X = pd.read_parquet("data/features.parquet")
    # y = pd.read_parquet("data/target.parquet")["target"]
    # or database query, etc.

    return X, y


def create_pipeline() -> Pipeline:
    """Create modeling pipeline with preprocessing + model"""
    return Pipeline([
        ("scaler", StandardScaler()),               # good practice even for trees in many cases
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"                 # useful when classes are imbalanced
        ))
    ])


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, Dict[str, Any]]:
    """Perform randomized search for good hyperparameters"""
    pipeline = create_pipeline()

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 400, 500],
        "classifier__max_depth": [None, 10, 15, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5, 0.7],
        "classifier__bootstrap": [True, False]
    }

    logger.info("Starting RandomizedSearchCV...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM_SEARCH,
        cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",                     # good choice for imbalanced classification
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    logger.info(f"RandomizedSearchCV completed in {elapsed:.1f} seconds")
    logger.info(f"Best CV f1_macro: {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")

    return search.best_estimator_, search.best_params_


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    logger.info(f"Test Accuracy      : {acc:.4f}")
    logger.info(f"Test Macro Precision: {precision:.4f}")
    logger.info(f"Test Macro Recall   : {recall:.4f}")
    logger.info(f"Test Macro F1       : {f1:.4f}")

    # Detailed report
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.named_steps["classifier"].classes_
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix plot")


def main():
    # 1. Load data
    X, y = load_and_prepare_data()

    # 2. Train / Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Hyperparameter tuning + best model
    best_model, best_params = hyperparameter_tuning(X_train, y_train)

    # 4. Final evaluation
    evaluate_model(best_model, X_test, y_test)

    # 5. Save model (you would normally use joblib or pickle)
    import joblib
    joblib.dump(best_model, MODEL_DIR / "best_random_forest_pipeline.joblib", compress=3)
    logger.info("Model saved successfully")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()