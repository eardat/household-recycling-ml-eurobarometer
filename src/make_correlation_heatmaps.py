"""Generate Cramer's V and Spearman correlation heatmaps for both datasets.

The survey predictors are mostly categorical or ordinal, so two complementary
views are produced for each dataset:

  * Cramer's V  - strength of association between categorical variables
                  (one-hot dummies are first rebuilt into their original
                  multi-category form so each variable appears once).
  * Spearman    - monotonic relationship between the ordered (ordinal /
                  continuous) variables and the target.

Run from the repository root:
    python -m src.make_correlation_heatmaps
"""
from pathlib import Path

import pandas as pd

from src.common.correlation import (
    cramers_v_matrix,
    spearman_matrix,
    reconstruct_from_dummies,
    plot_cramers_v,
    plot_spearman,
)
from src.eb682.prepare_eb682 import build_features as build_eb682_features

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
FEB316 = ROOT / "data" / "processed" / "FEB316" / "eurobarometer_processed.csv"
EB682 = ROOT / "data" / "processed" / "EB68_2" / "eb682_processed.csv"

# Display names so the figures read cleanly and match the thesis terminology.
LABELS = {
    "target_recycles": "recycler (target)",
    "target_recycle_rate": "country_recycle_rate",
}


def _relabel(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.rename(index=LABELS, columns=LABELS)


def process_feb316() -> None:
    df = pd.read_csv(FEB316)
    df["activity"] = reconstruct_from_dummies(df, "activity_", "activity")
    df["occupation_group"] = reconstruct_from_dummies(df, "d4_grouped_", "occupation_group")

    cramer_cols = [
        "gender", "age_group", "area_type", "d3_edu",
        "activity", "occupation_group", "country", "target_recycles",
    ]
    spearman_cols = [
        "d3_edu", "age_group", "area_type", "target_recycle_rate", "target_recycles",
    ]

    cv = _relabel(cramers_v_matrix(df, cramer_cols))
    sp = _relabel(spearman_matrix(df, spearman_cols))
    plot_cramers_v(cv, "FEB316 - Cramer's V association matrix",
                   FIGURES / "cramersv_FEB316.png")
    plot_spearman(sp, "FEB316 - Spearman rank correlation",
                  FIGURES / "spearman_FEB316.png")
    print("[FEB316] Cramer's V vs target (recycler):")
    print(cv["recycler (target)"].drop("recycler (target)").sort_values(ascending=False).round(3).to_string())
    print("\n[FEB316] Spearman vs target (recycler):")
    print(sp["recycler (target)"].drop("recycler (target)").sort_values(key=abs, ascending=False).round(3).to_string())


def process_eb682() -> None:
    # The static eb682_processed.csv is stale: it carries an "eco_mindset"
    # column that the modelling pipeline explicitly excludes. Rebuild the
    # features from the raw survey so the heatmaps use exactly the predictors
    # the models and SHAP analysis use (notably green_behavior_score).
    X, y = build_eb682_features()
    df = X.copy()
    df["target_recycles"] = y.to_numpy()

    cramer_cols = [
        "gender", "age_group", "area_type", "d3_edu", "green_behavior_score",
        "env_importance", "eco_purchase", "household_size", "wealth_index",
        "occupation_group", "target_recycles",
    ]
    spearman_cols = [
        "d3_edu", "age_group", "area_type", "wealth_index", "env_importance",
        "eco_purchase", "household_size", "green_behavior_score", "target_recycles",
    ]

    cv = _relabel(cramers_v_matrix(df, cramer_cols))
    sp = _relabel(spearman_matrix(df, spearman_cols))
    plot_cramers_v(cv, "EB68.2 - Cramer's V association matrix",
                   FIGURES / "cramersv_EB68_2.png")
    plot_spearman(sp, "EB68.2 - Spearman rank correlation",
                  FIGURES / "spearman_EB68_2.png")
    print("\n[EB68.2] Cramer's V vs target (recycler):")
    print(cv["recycler (target)"].drop("recycler (target)").sort_values(ascending=False).round(3).to_string())
    print("\n[EB68.2] Spearman vs target (recycler):")
    print(sp["recycler (target)"].drop("recycler (target)").sort_values(key=abs, ascending=False).round(3).to_string())


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    process_feb316()
    process_eb682()
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
