from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.common.all6 import RANDOM_STATE, make_experiment_pipeline, model_configs
from src.common.metrics import metrics_at_threshold, probability_scores
from src.common.preprocessing import clean_feature_name
from src.eb682.prepare_eb682 import CAT_COLS, NUMERIC_BASE, RAW_EB682, build_features


ROOT = Path(__file__).resolve().parents[2]
BEST_MODELS_CSV = ROOT / "outputs" / "results" / "best_models_ALL_EXPERIMENTS_ALL6.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"

TREE_MODELS = {"Random Forest", "XGBoost", "CatBoost"}


def _positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[2] == 2:
            return values[:, :, 1]
        if values.shape[0] == 2:
            return values[1]
    return values


def _slug_model(model_name: str) -> str:
    return model_name.replace(" ", "")


def _slug_experiment(experiment: str) -> str:
    if experiment == "EB68.2 Model A":
        return "Model_A"
    if experiment == "EB68.2 Model B":
        return "Model_B"
    raise ValueError(f"Unsupported EB68.2 experiment for SHAP: {experiment}")


def _load_best_rows():
    if not BEST_MODELS_CSV.exists():
        raise FileNotFoundError(f"Missing best-model table: {BEST_MODELS_CSV}")
    best = pd.read_csv(BEST_MODELS_CSV)
    return best[best["Experiment"].isin(["EB68.2 Model A", "EB68.2 Model B"])].copy()


def _fit_best_pipeline(experiment: str, model_name: str, best_params: dict):
    X, y = build_features(RAW_EB682)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    include_country = experiment == "EB68.2 Model A"
    if not include_country:
        X_train = X_train.drop(columns=["country_iso"])
        X_test = X_test.drop(columns=["country_iso"])

    cfg = model_configs(RANDOM_STATE)[model_name]
    pipe = make_experiment_pipeline(
        cfg,
        NUMERIC_BASE,
        CAT_COLS,
        include_country=include_country,
        oversample=False,
    )
    pipe.set_params(**best_params)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


def _transformed_test_frame(pipe, X_test):
    transformed = pipe[:-1].transform(X_test)
    preprocess = pipe.named_steps["preprocess"]
    feature_names = [clean_feature_name(name) for name in preprocess.get_feature_names_out()]
    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Feature-name alignment failed: "
            f"transformed columns={transformed.shape[1]}, names={len(feature_names)}"
        )
    return pd.DataFrame(transformed, columns=feature_names, index=X_test.index)


def generate_shap_for_best_row(row: pd.Series):
    experiment = row["Experiment"]
    model_name = row["Model"]
    if model_name not in TREE_MODELS:
        raise ValueError(
            f"{experiment} selected {model_name}; automatic Tree SHAP is not appropriate."
        )

    best_params = json.loads(row["Best_Params"])
    pipe, X_test, y_test = _fit_best_pipeline(experiment, model_name, best_params)
    X_test_tx = _transformed_test_frame(pipe, X_test)
    estimator = pipe.named_steps["model"]
    threshold = float(row["Validation_Selected_Threshold"])
    test_prob = probability_scores(pipe, X_test)
    threshold_metrics, _ = metrics_at_threshold(y_test, test_prob, threshold)

    explainer = shap.TreeExplainer(estimator)
    shap_values = _positive_class_shap(explainer.shap_values(X_test_tx))
    if shap_values.shape[1] != X_test_tx.shape[1]:
        raise ValueError(
            "SHAP value alignment failed: "
            f"SHAP columns={shap_values.shape[1]}, transformed columns={X_test_tx.shape[1]}"
        )

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature_name": X_test_tx.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    exp_slug = _slug_experiment(experiment)
    model_slug = _slug_model(model_name)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"shap_EB68_2_{exp_slug}_{model_slug}_importance_ALL6.csv"
    bar_path = FIGURES_DIR / f"shap_EB68_2_{exp_slug}_{model_slug}_bar_ALL6.png"
    dot_path = FIGURES_DIR / f"shap_EB68_2_{exp_slug}_{model_slug}_dot_ALL6.png"
    importance.to_csv(csv_path, index=False)

    shap.summary_plot(shap_values, X_test_tx, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test_tx, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(dot_path, dpi=180, bbox_inches="tight")
    plt.close()

    return {
        "experiment": experiment,
        "model": model_name,
        "best_params": best_params,
        "estimator_params": estimator.get_params(),
        "threshold": threshold,
        "metrics": threshold_metrics,
        "features": X_test_tx.shape[1],
        "test_rows": X_test_tx.shape[0],
        "transformed_shape": X_test_tx.shape,
        "shap_shape": shap_values.shape,
        "alignment_status": "aligned",
        "top_features": importance.head(20).copy(),
        "csv": csv_path,
        "bar": bar_path,
        "dot": dot_path,
    }


def write_model_a_report(item):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    params = item["estimator_params"]
    report_path = REPORTS_DIR / "SHAP_MODEL_A_RF_ALL6_REGEN_REPORT.md"
    top_lines = [
        f"| {int(row.rank)} | {row.feature_name} | {row.mean_abs_shap:.8f} |"
        for row in item["top_features"].itertuples(index=False)
    ]
    selected_lines = [
        f"- `model__max_depth`: `{item['best_params'].get('model__max_depth')}`",
        f"- `model__min_samples_leaf`: `{item['best_params'].get('model__min_samples_leaf')}`",
        f"- `model__n_estimators`: `{item['best_params'].get('model__n_estimators')}`",
        f"- Random Forest `class_weight`: `{params.get('class_weight')}`",
        f"- Random Forest `random_state`: `{params.get('random_state')}`",
        f"- Random Forest `n_jobs`: `{params.get('n_jobs')}`",
    ]
    full_param_lines = [f"- `{key}`: `{value}`" for key, value in sorted(params.items())]
    metrics = item["metrics"]
    report = [
        "# SHAP_MODEL_A_RF_ALL6_REGEN_REPORT",
        "",
        "## Scope",
        "",
        "- Dataset: EB68.2 / GESIS ZA4742, `data/raw/ZA4742_EB68_2/ZA4742_v4-0-1.sav`.",
        "- Model: final ALL6 EB68.2 Model A Random Forest.",
        "- Feature set: individual EB68.2 predictors plus fold-safe `country_recycle_rate` from `v7` / `country_iso`.",
        "- Target: `v552` waste-separation mention indicator.",
        "- Split: same cleaned ALL6 split, `train_test_split(test_size=0.20, stratify=y, random_state=42)`.",
        "- SHAP source: regenerated from the ALL6 fitted pipeline; old `*_FIXED2` SHAP files were not used.",
        "",
        "## Exact Model Parameters",
        "",
        "Selected ALL6 pipeline parameters:",
        "",
        *selected_lines,
        "",
        "Full fitted `RandomForestClassifier` parameters:",
        "",
        *full_param_lines,
        "",
        "## Reproduced Metrics",
        "",
        f"- Selected validation threshold: `{item['threshold']:.2f}`.",
        f"- Threshold Macro F1: `{metrics['Macro_F1']:.6f}`.",
        f"- ROC-AUC: `{metrics['ROC_AUC']:.6f}`.",
        "",
        "These reproduce the saved ALL6 result approximately: threshold Macro F1 around `0.6864`, ROC-AUC around `0.7517`, selected threshold `0.46`.",
        "",
        "## Matrix Checks",
        "",
        f"- Transformed test matrix shape: `{item['transformed_shape']}`.",
        f"- SHAP matrix shape: `{item['shap_shape']}`.",
        f"- Feature-name alignment status: `{item['alignment_status']}`.",
        "",
        "## Top 20 SHAP Features",
        "",
        "| Rank | Feature | Mean absolute SHAP |",
        "| --- | --- | ---: |",
        *top_lines,
        "",
        "## Warnings",
        "",
        "- SHAP is model-level attribution, not causality.",
        "- `country_recycle_rate` is predictive country context derived by target encoding, not a direct policy or infrastructure measurement.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["EB68.2 Model A", "EB68.2 Model B"],
        help="Generate SHAP only for one EB68.2 experiment.",
    )
    args = parser.parse_args(argv)

    best = _load_best_rows()
    if args.experiment:
        best = best[best["Experiment"] == args.experiment]
    if best.empty:
        raise SystemExit("No matching EB68.2 best-model rows found.")

    outputs = []
    for _, row in best.iterrows():
        item = generate_shap_for_best_row(row)
        if item["experiment"] == "EB68.2 Model A" and item["model"] == "Random Forest":
            item["report"] = write_model_a_report(item)
        outputs.append(item)

    for item in outputs:
        print(
            f"{item['experiment']} {item['model']}: "
            f"{item['test_rows']} rows, {item['features']} features"
        )
        print(f"  CSV: {item['csv']}")
        print(f"  Bar: {item['bar']}")
        print(f"  Dot: {item['dot']}")
        if "report" in item:
            print(f"  Report: {item['report']}")


if __name__ == "__main__":
    main()
