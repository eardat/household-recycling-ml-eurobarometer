from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.common.metrics import probability_scores, metrics_at_threshold
from src.common.plotting import plot_confusions
from src.common.preprocessing import CountryTargetEncoder, make_preprocessor
from src.common.thresholding import choose_threshold


RANDOM_STATE = 42


def model_configs(random_state=RANDOM_STATE):
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced"),
            "params": {"model__C": [0.1, 1.0, 10.0]},
            "scale": True,
        },
        "SVM": {
            "estimator": CalibratedClassifierCV(
                estimator=LinearSVC(random_state=random_state, max_iter=5000, class_weight="balanced"),
                cv=3,
            ),
            "params": {"model__estimator__C": [0.1, 1.0]},
            "scale": True,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced"),
            "params": {"model__n_estimators": [200], "model__max_depth": [10, 20], "model__min_samples_leaf": [1, 5]},
            "scale": False,
        },
        "XGBoost": {
            "estimator": XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1, tree_method="hist"),
            "params": {"model__n_estimators": [150], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
        },
        "CatBoost": {
            "estimator": CatBoostClassifier(random_seed=random_state, verbose=0, allow_writing_files=False, eval_metric="F1"),
            "params": {"model__iterations": [200], "model__depth": [4, 6], "model__learning_rate": [0.05]},
            "scale": False,
        },
        "ANN": {
            "estimator": MLPClassifier(random_state=random_state, max_iter=300, early_stopping=True),
            "params": {"model__hidden_layer_sizes": [(64, 32)], "model__learning_rate_init": [0.001]},
            "scale": True,
        },
    }


def make_experiment_pipeline(cfg, numeric_cols, cat_cols, include_country=False, oversample=False):
    steps = []
    if include_country:
        steps.append(("country_encoder", CountryTargetEncoder("country_iso")))
        numeric_cols = list(numeric_cols) + ["country_recycle_rate"]
    steps.append(("preprocess", make_preprocessor(numeric_cols, cat_cols, cfg["scale"])))
    if oversample:
        steps.append(("oversample", RandomOverSampler(random_state=RANDOM_STATE)))
        return ImbPipeline(steps + [("model", cfg["estimator"])])
    return Pipeline(steps + [("model", cfg["estimator"])])


def run_experiment(
    *,
    dataset_label,
    experiment_label,
    X,
    y,
    numeric_cols,
    cat_cols,
    include_country,
    oversample,
    output_csv,
    random_state=RANDOM_STATE,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    Ttr, Tval, ytr, yval = train_test_split(
        X_train, y_train, test_size=0.20, random_state=random_state, stratify=y_train
    )
    if not include_country and "country_iso" in X_train.columns:
        X_train = X_train.drop(columns=["country_iso"])
        X_test = X_test.drop(columns=["country_iso"])
        Ttr = Ttr.drop(columns=["country_iso"])
        Tval = Tval.drop(columns=["country_iso"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows, models, probs, preds = [], {}, {}, {}
    for name, cfg in model_configs(random_state).items():
        pipe = make_experiment_pipeline(cfg, numeric_cols, cat_cols, include_country=include_country, oversample=oversample)
        gs = GridSearchCV(pipe, cfg["params"], cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"[{experiment_label}] fitting {name}", flush=True)
        gs.fit(X_train, y_train)
        test_prob = probability_scores(gs.best_estimator_, X_test)
        default_metrics, _ = metrics_at_threshold(y_test, test_prob, 0.50)
        thresh_est = clone(gs.best_estimator_)
        thresh_est.fit(Ttr, ytr)
        val_prob = probability_scores(thresh_est, Tval)
        threshold, val_f1 = choose_threshold(yval, val_prob)
        threshold_metrics, threshold_pred = metrics_at_threshold(y_test, test_prob, threshold)
        row = {
            "Dataset": dataset_label,
            "Experiment": experiment_label,
            "Model": name,
            "CV_Macro_F1": gs.best_score_,
            "Best_Params": json.dumps(gs.best_params_, sort_keys=True),
            "Validation_Selected_Threshold": threshold,
            "Validation_Macro_F1_at_Selected_Threshold": val_f1,
            "Train_Size": len(X_train),
            "Test_Size": len(X_test),
        }
        row.update({f"Default_{k}": v for k, v in default_metrics.items()})
        row.update({f"Threshold_{k}": v for k, v in threshold_metrics.items()})
        rows.append(row)
        models[name] = gs.best_estimator_
        probs[name] = test_prob
        preds[name] = threshold_pred
    result = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result, {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "models": models, "probs": probs, "preds": preds}


def save_best_figures(result_df, run_info, confusion_path, roc_path):
    best = result_df.sort_values("Threshold_Macro_F1", ascending=False).iloc[0]
    name = best["Model"]
    cm = confusion_matrix(run_info["y_test"], run_info["preds"][name])
    plot_confusions({name: cm}, f"Best model confusion matrix - {best['Experiment']}", confusion_path)

    y_test = run_info["y_test"]
    prob = run_info["probs"][name]
    fpr, tpr, _ = roc_curve(y_test, prob)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"Best model ROC - {best['Experiment']}", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=180)
    plt.close(fig)
    return best
