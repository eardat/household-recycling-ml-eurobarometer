from pathlib import Path

from src.common.all6 import run_experiment, save_best_figures
from src.eb682.prepare_eb682 import build_features, NUMERIC_BASE, CAT_COLS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"


def main():
    X, y = build_features()
    result, info = run_experiment(
        dataset_label="Eurobarometer 68.2 / GESIS ZA4742",
        experiment_label="EB68.2 Model A",
        X=X,
        y=y,
        numeric_cols=NUMERIC_BASE,
        cat_cols=CAT_COLS,
        include_country=True,
        oversample=False,
        output_csv=RESULTS / "results_EB68_2_Model_A_ALL6.csv",
    )
    save_best_figures(
        result,
        info,
        FIGURES / "confusion_EB68_2_Model_A_best_ALL6.png",
        FIGURES / "roc_EB68_2_Model_A_best_ALL6.png",
    )
    return result


if __name__ == "__main__":
    main()
