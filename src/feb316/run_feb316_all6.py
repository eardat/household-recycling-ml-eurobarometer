from pathlib import Path

from src.common.all6 import run_experiment, save_best_figures
from src.feb316.prepare_feb316 import build_features, NUMERIC_BASE, CAT_COLS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"


def main():
    X, y = build_features()
    result, info = run_experiment(
        dataset_label="Flash Eurobarometer 316 / GESIS ZA5474",
        experiment_label="FEB316",
        X=X,
        y=y,
        numeric_cols=NUMERIC_BASE,
        cat_cols=CAT_COLS,
        include_country=True,
        oversample=True,
        output_csv=RESULTS / "results_FEB316_ALL6.csv",
    )
    save_best_figures(
        result,
        info,
        FIGURES / "confusion_FEB316_best_ALL6.png",
        FIGURES / "roc_FEB316_best_ALL6.png",
    )
    return result


if __name__ == "__main__":
    main()
