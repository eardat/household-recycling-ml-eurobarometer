# Recycling Behaviour Machine Learning Thesis

This repository contains the cleaned, reproducible code path for a graduation thesis on recycling behaviour prediction using Eurobarometer survey data.

## Datasets

- **Flash Eurobarometer 316 / GESIS ZA5474**: separate benchmark experiment. It does not use EB68.2 Model A/Model B terminology.
- **Eurobarometer 68.2 / GESIS ZA4742**: main corrected experiment.
  - Model A: individual predictors plus fold-safe `country_recycle_rate` from `v7` / `country_iso`.
  - Model B: individual predictors only.
  - Final ALL6 clean feature set excludes `eco_mindset`, `green_energy`, `green_food`, and `green_water`, while retaining `green_behavior_score`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Final Commands

Running Model A/B retrains models and may take time.

```bash
python -m src.eb682.run_model_a_all6
python -m src.eb682.run_model_b_all6
python -m src.feb316.run_feb316_all6
python -m src.make_all_results
```

## Expected Final Outputs

Final result and report files:

- `outputs/results/results_FEB316_ALL6.csv`
- `outputs/results/results_EB68_2_Model_A_ALL6.csv`
- `outputs/results/results_EB68_2_Model_B_ALL6.csv`
- `outputs/results/model_comparison_ALL_EXPERIMENTS_ALL6.csv`
- `outputs/results/best_models_ALL_EXPERIMENTS_ALL6.csv`
- `outputs/reports/ALL_EXPERIMENTS_ALL6_UPDATE_REPORT.md`
- `outputs/results/shap_EB68_2_Model_A_RandomForest_importance_ALL6.csv`
- `outputs/results/shap_EB68_2_Model_B_RandomForest_importance_ALL6.csv`

Final thesis-ready figure files:

- `outputs/figures/confusion_FEB316_best_ALL6.png`
- `outputs/figures/roc_FEB316_best_ALL6.png`
- `outputs/figures/confusion_EB68_2_Model_A_best_ALL6.png`
- `outputs/figures/roc_EB68_2_Model_A_best_ALL6.png`
- `outputs/figures/confusion_EB68_2_Model_B_best_ALL6.png`
- `outputs/figures/roc_EB68_2_Model_B_best_ALL6.png`
- `outputs/figures/shap_EB68_2_Model_A_RandomForest_bar_ALL6.png`
- `outputs/figures/shap_EB68_2_Model_A_RandomForest_dot_ALL6.png`
- `outputs/figures/shap_EB68_2_Model_B_RandomForest_bar_ALL6.png`
- `outputs/figures/shap_EB68_2_Model_B_RandomForest_dot_ALL6.png`

The same final thesis-ready figures are copied to `thesis/final_figures/`.

## Archive

`archive_old/` contains previous exploratory notebooks, old scripts, old result tables, old figures, and superseded reports. Nothing was permanently deleted during cleanup.
