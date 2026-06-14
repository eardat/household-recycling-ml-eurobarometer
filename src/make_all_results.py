from pathlib import Path
import platform

import pandas as pd
import sklearn
import numpy
import pandas

try:
    import xgboost
except Exception:
    xgboost = None
try:
    import catboost
except Exception:
    catboost = None
try:
    import shap
except Exception:
    shap = None

from src.common.all6 import model_configs
from src.feb316.run_feb316_all6 import main as run_feb316
from src.eb682.run_model_a_all6 import main as run_model_a
from src.eb682.run_model_b_all6 import main as run_model_b

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
REPORTS = ROOT / "outputs" / "reports"


def _fmt_table(df, include_experiment=False):
    cols = []
    if include_experiment:
        cols.append("Experiment")
    cols += [
        "Model", "CV_Macro_F1", "Default_Accuracy", "Default_Macro_Precision",
        "Default_Macro_Recall", "Default_Macro_F1", "Default_ROC_AUC",
        "Validation_Selected_Threshold", "Threshold_Macro_F1", "Threshold_ROC_AUC",
    ]
    out = "| " + " | ".join(cols) + " |\n"
    out += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            val = row[col]
            values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        out += "| " + " | ".join(values) + " |\n"
    return out


def _versions():
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": getattr(xgboost, "__version__", "not available"),
        "catboost": getattr(catboost, "__version__", "not available"),
        "shap": getattr(shap, "__version__", "not available"),
    }


def _write_report(feb, model_a, model_b, combined, best):
    REPORTS.mkdir(parents=True, exist_ok=True)
    grids = model_configs()
    grid_lines = []
    for name, cfg in grids.items():
        grid_lines.append(f"- {name}: `{cfg['params']}`")

    previous_best_a = "Random Forest"
    previous_best_b = "CatBoost"
    best_a = best.loc[best["Experiment"] == "EB68.2 Model A", "Model"].iloc[0]
    best_b = best.loc[best["Experiment"] == "EB68.2 Model B", "Model"].iloc[0]
    shap_change = []
    if best_a != previous_best_a:
        shap_change.append(f"Model A SHAP should change because previous SHAP best model was {previous_best_a}, now {best_a}.")
    else:
        shap_change.append(f"Model A SHAP model remains {best_a}; no SHAP regeneration required by model selection.")
    if best_b != previous_best_b:
        shap_change.append(f"Model B SHAP should change because previous SHAP best model was {previous_best_b}, now {best_b}.")
    else:
        shap_change.append(f"Model B SHAP model remains {best_b}; no SHAP regeneration required by model selection.")

    shap_artifacts = []
    model_b_rf_paths = [
        "outputs/results/shap_EB68_2_Model_B_RandomForest_importance_ALL6.csv",
        "outputs/figures/shap_EB68_2_Model_B_RandomForest_bar_ALL6.png",
        "outputs/figures/shap_EB68_2_Model_B_RandomForest_dot_ALL6.png",
    ]
    if best_b == "Random Forest" and all((ROOT / p).exists() for p in model_b_rf_paths):
        shap_artifacts += [
            "- EB68.2 Model B changed from CatBoost to Random Forest under ALL6; regenerated Random Forest SHAP with aligned transformed test features.",
            *[f"- `{p}`" for p in model_b_rf_paths],
        ]
    elif best_b == "Random Forest":
        shap_artifacts.append("- EB68.2 Model B changed to Random Forest; run `python -m src.eb682.shap_best_models --experiment \"EB68.2 Model B\"` to regenerate matching SHAP artifacts.")
    if best_a == previous_best_a:
        shap_artifacts.append("- EB68.2 Model A remains Random Forest, matching the existing best-model SHAP family; no additional ALL6 SHAP regeneration was required by model selection.")

    svm_ann_implications = []
    for _, row in best.iterrows():
        if row["Model"] in {"SVM", "ANN"}:
            svm_ann_implications.append(
                f"- {row['Experiment']}: best model is {row['Model']}; avoid automatic Tree SHAP. Use Linear SHAP for linear SVM, Kernel SHAP on a sample, permutation importance, or no SHAP depending on thesis scope."
            )
    if not svm_ann_implications:
        svm_ann_implications.append("- No setting selected SVM or ANN as best, so tree-based SHAP remains appropriate for EB68.2 best models when the best models are tree ensembles.")

    versions = _versions()
    version_lines = [f"- {k}: {v}" for k, v in versions.items()]

    report = [
        "# ALL_EXPERIMENTS_ALL6_UPDATE_REPORT",
        "",
        "## Exact Datasets Used",
        "",
        "- FEB316: `data/raw/ZA5474_FEB316/ZA5474_v1-0-0.xlsx`",
        "- EB68.2: `data/raw/ZA4742_EB68_2/ZA4742_v4-0-1.sav`",
        "",
        "## Train/Test Sizes",
        "",
        f"- FEB316: train {int(feb['Train_Size'].iloc[0])}, test {int(feb['Test_Size'].iloc[0])}",
        f"- EB68.2 Model A: train {int(model_a['Train_Size'].iloc[0])}, test {int(model_a['Test_Size'].iloc[0])}",
        f"- EB68.2 Model B: train {int(model_b['Train_Size'].iloc[0])}, test {int(model_b['Test_Size'].iloc[0])}",
        "",
        "## Target Definitions",
        "",
        "- FEB316 target: `q2`, mapped `No = 0`, `Yes = 1`.",
        "- EB68.2 target: `v552`, `0 = Not mentioned`, `1 = Mentioned` for QF13 waste separation.",
        "",
        "## Final Feature Sets",
        "",
        "- FEB316: `d3_edu`, `gender`, `age_group`, `area_type`, `activity`, `occupation_group`, and fold-safe `country_recycle_rate` from country.",
        "- EB68.2 Model A: `gender`, `age_group`, `area_type`, `d3_edu`, `green_behavior_score`, `env_importance`, `eco_purchase`, `household_size`, `wealth_index`, `occupation_group`, plus fold-safe `country_recycle_rate` from `v7` / `country_iso`.",
        "- EB68.2 Model B: same individual EB68.2 predictors as Model A, but without `country_iso` or `country_recycle_rate`.",
        "- EB68.2 excludes `eco_mindset`, `green_energy`, `green_food`, and `green_water` as separate predictors.",
        "",
        "## Algorithms Included",
        "",
        "All three settings include the same six algorithms: Logistic Regression, SVM, Random Forest, XGBoost, CatBoost, and ANN.",
        "",
        "## Hyperparameter Grids",
        "",
        *grid_lines,
        "",
        "## CV, Thresholding, Seeds",
        "",
        "- CV folds: 5-fold StratifiedKFold with shuffle=True and random_state=42.",
        "- Threshold strategy: validation-only split from training data, `test_size=0.20`, random_state=42, stratified.",
        "- Held-out test targets are not used for threshold selection.",
        "- FEB316 oversampling: RandomOverSampler inside the imblearn pipeline, therefore inside CV/training-fold-only logic. Test set is never oversampled.",
        "- EB68.2 oversampling: none.",
        "- Random seed: 42 wherever applicable.",
        "",
        "## Library Versions",
        "",
        *version_lines,
        "",
        "## FEB316 Results",
        "",
        _fmt_table(feb),
        "## EB68.2 Model A Results",
        "",
        _fmt_table(model_a),
        "## EB68.2 Model B Results",
        "",
        _fmt_table(model_b),
        "## Best Models By Validation-Selected Test Macro F1",
        "",
        _fmt_table(best, include_experiment=True),
        "## Best Model Changes Compared With Previous Thesis Version",
        "",
        "- Previous cleaned EB68.2 Model A best model was Random Forest; compare with table above.",
        "- Previous cleaned EB68.2 Model B best model was CatBoost; compare with table above.",
        "- FEB316 previous benchmark should be replaced by this standardized six-algorithm table.",
        "",
        "## SHAP Model Selection",
        "",
        *[f"- {line}" for line in shap_change],
        "",
        "## SHAP Artifacts",
        "",
        *shap_artifacts,
        "",
        "## SVM / ANN Explainability Implications",
        "",
        *svm_ann_implications,
        "",
        "## Warnings",
        "",
        "- SHAP and feature importance are model-level attribution methods, not causal evidence.",
        "- `country_recycle_rate` is target-encoded aggregate predictive context, not a direct infrastructure or policy measurement.",
        "- The random train/test split evaluates known-country respondent prediction, not unseen-country generalization.",
        "- FEB316 uses oversampling only inside training/CV folds; held-out test rows remain untouched.",
        "- EB68.2 target interpretation must remain tied to `v552` waste-separation mention status.",
    ]
    (REPORTS / "ALL_EXPERIMENTS_ALL6_UPDATE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    feb = run_feb316()
    model_a = run_model_a()
    model_b = run_model_b()
    combined = pd.concat([feb, model_a, model_b], ignore_index=True)
    combined.to_csv(RESULTS / "model_comparison_ALL_EXPERIMENTS_ALL6.csv", index=False)
    best = combined.sort_values("Threshold_Macro_F1", ascending=False).groupby("Experiment", as_index=False).first()
    best.to_csv(RESULTS / "best_models_ALL_EXPERIMENTS_ALL6.csv", index=False)
    _write_report(feb, model_a, model_b, combined, best)
    return combined, best


if __name__ == "__main__":
    main()
