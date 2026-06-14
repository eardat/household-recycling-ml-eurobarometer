# ALL_EXPERIMENTS_ALL6_UPDATE_REPORT

## Exact Datasets Used

- FEB316: `data/raw/ZA5474_FEB316/ZA5474_v1-0-0.xlsx`
- EB68.2: `data/raw/ZA4742_EB68_2/ZA4742_v4-0-1.sav`

## Train/Test Sizes

- FEB316: train 21684, test 5422
- EB68.2 Model A: train 21384, test 5346
- EB68.2 Model B: train 21384, test 5346

## Target Definitions

- FEB316 target: `q2`, mapped `No = 0`, `Yes = 1`.
- EB68.2 target: `v552`, `0 = Not mentioned`, `1 = Mentioned` for QF13 waste separation.

## Final Feature Sets

- FEB316: `d3_edu`, `gender`, `age_group`, `area_type`, `activity`, `occupation_group`, and fold-safe `country_recycle_rate` from country.
- EB68.2 Model A: `gender`, `age_group`, `area_type`, `d3_edu`, `green_behavior_score`, `env_importance`, `eco_purchase`, `household_size`, `wealth_index`, `occupation_group`, plus fold-safe `country_recycle_rate` from `v7` / `country_iso`.
- EB68.2 Model B: same individual EB68.2 predictors as Model A, but without `country_iso` or `country_recycle_rate`.
- EB68.2 excludes `eco_mindset`, `green_energy`, `green_food`, and `green_water` as separate predictors.

## Algorithms Included

All three settings include the same six algorithms: Logistic Regression, SVM, Random Forest, XGBoost, CatBoost, and ANN.

## Hyperparameter Grids

- Logistic Regression: `{'model__C': [0.1, 1.0, 10.0]}`
- SVM: `{'model__estimator__C': [0.1, 1.0]}`
- Random Forest: `{'model__n_estimators': [200], 'model__max_depth': [10, 20], 'model__min_samples_leaf': [1, 5]}`
- XGBoost: `{'model__n_estimators': [150], 'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1]}`
- CatBoost: `{'model__iterations': [200], 'model__depth': [4, 6], 'model__learning_rate': [0.05]}`
- ANN: `{'model__hidden_layer_sizes': [(64, 32)], 'model__learning_rate_init': [0.001]}`

## CV, Thresholding, Seeds

- CV folds: 5-fold StratifiedKFold with shuffle=True and random_state=42.
- Threshold strategy: validation-only split from training data, `test_size=0.20`, random_state=42, stratified.
- Held-out test targets are not used for threshold selection.
- FEB316 oversampling: RandomOverSampler inside the imblearn pipeline, therefore inside CV/training-fold-only logic. Test set is never oversampled.
- EB68.2 oversampling: none.
- Random seed: 42 wherever applicable.

## Library Versions

- python: 3.11.15
- numpy: 2.4.6
- pandas: 3.0.3
- scikit-learn: 1.9.0
- xgboost: 3.2.0
- catboost: 1.2.10
- shap: 0.51.0

## FEB316 Results

| Model | CV_Macro_F1 | Default_Accuracy | Default_Macro_Precision | Default_Macro_Recall | Default_Macro_F1 | Default_ROC_AUC | Validation_Selected_Threshold | Threshold_Macro_F1 | Threshold_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6266 | 0.7604 | 0.6222 | 0.7184 | 0.6337 | 0.7801 | 0.3000 | 0.6684 | 0.7801 |
| SVM | 0.6259 | 0.7610 | 0.6226 | 0.7187 | 0.6342 | 0.7804 | 0.3100 | 0.6711 | 0.7804 |
| Random Forest | 0.6093 | 0.7311 | 0.6097 | 0.7093 | 0.6120 | 0.7726 | 0.3400 | 0.6617 | 0.7726 |
| XGBoost | 0.6092 | 0.7174 | 0.6093 | 0.7159 | 0.6062 | 0.7826 | 0.3100 | 0.6712 | 0.7826 |
| CatBoost | 0.6069 | 0.7254 | 0.6117 | 0.7175 | 0.6116 | 0.7817 | 0.3100 | 0.6714 | 0.7817 |
| ANN | 0.6022 | 0.7222 | 0.6095 | 0.7139 | 0.6083 | 0.7697 | 0.3400 | 0.6571 | 0.7697 |

## EB68.2 Model A Results

| Model | CV_Macro_F1 | Default_Accuracy | Default_Macro_Precision | Default_Macro_Recall | Default_Macro_F1 | Default_ROC_AUC | Validation_Selected_Threshold | Threshold_Macro_F1 | Threshold_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6843 | 0.6828 | 0.6795 | 0.6805 | 0.6799 | 0.7435 | 0.4600 | 0.6761 | 0.7435 |
| SVM | 0.6791 | 0.6829 | 0.6797 | 0.6730 | 0.6741 | 0.7438 | 0.5500 | 0.6790 | 0.7438 |
| Random Forest | 0.6857 | 0.6900 | 0.6870 | 0.6881 | 0.6874 | 0.7517 | 0.4600 | 0.6864 | 0.7517 |
| XGBoost | 0.6869 | 0.6929 | 0.6900 | 0.6833 | 0.6845 | 0.7519 | 0.5000 | 0.6845 | 0.7519 |
| CatBoost | 0.6827 | 0.6880 | 0.6850 | 0.6782 | 0.6794 | 0.7496 | 0.5200 | 0.6821 | 0.7496 |
| ANN | 0.6777 | 0.6863 | 0.6828 | 0.6773 | 0.6785 | 0.7406 | 0.5700 | 0.6764 | 0.7406 |

## EB68.2 Model B Results

| Model | CV_Macro_F1 | Default_Accuracy | Default_Macro_Precision | Default_Macro_Recall | Default_Macro_F1 | Default_ROC_AUC | Validation_Selected_Threshold | Threshold_Macro_F1 | Threshold_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6155 | 0.6008 | 0.6002 | 0.6014 | 0.5994 | 0.6475 | 0.4700 | 0.6013 | 0.6475 |
| SVM | 0.6016 | 0.6150 | 0.6084 | 0.6004 | 0.5990 | 0.6476 | 0.5200 | 0.5990 | 0.6476 |
| Random Forest | 0.6229 | 0.6188 | 0.6167 | 0.6179 | 0.6166 | 0.6634 | 0.5000 | 0.6166 | 0.6634 |
| XGBoost | 0.6181 | 0.6235 | 0.6172 | 0.6111 | 0.6108 | 0.6695 | 0.5100 | 0.6132 | 0.6695 |
| CatBoost | 0.6177 | 0.6197 | 0.6133 | 0.6069 | 0.6064 | 0.6677 | 0.5400 | 0.6147 | 0.6677 |
| ANN | 0.6094 | 0.6115 | 0.6046 | 0.6009 | 0.6009 | 0.6512 | 0.5000 | 0.6009 | 0.6512 |

## Best Models By Validation-Selected Test Macro F1

| Experiment | Model | CV_Macro_F1 | Default_Accuracy | Default_Macro_Precision | Default_Macro_Recall | Default_Macro_F1 | Default_ROC_AUC | Validation_Selected_Threshold | Threshold_Macro_F1 | Threshold_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EB68.2 Model A | Random Forest | 0.6857 | 0.6900 | 0.6870 | 0.6881 | 0.6874 | 0.7517 | 0.4600 | 0.6864 | 0.7517 |
| EB68.2 Model B | Random Forest | 0.6229 | 0.6188 | 0.6167 | 0.6179 | 0.6166 | 0.6634 | 0.5000 | 0.6166 | 0.6634 |
| FEB316 | CatBoost | 0.6069 | 0.7254 | 0.6117 | 0.7175 | 0.6116 | 0.7817 | 0.3100 | 0.6714 | 0.7817 |

## Best Model Changes Compared With Previous Thesis Version

- Previous cleaned EB68.2 Model A best model was Random Forest; compare with table above.
- Previous cleaned EB68.2 Model B best model was CatBoost; compare with table above.
- FEB316 previous benchmark should be replaced by this standardized six-algorithm table.

## SHAP Model Selection

- Model A SHAP model remains Random Forest; no SHAP regeneration required by model selection.
- Model B SHAP should change because previous SHAP best model was CatBoost, now Random Forest.

## SHAP Artifacts

- EB68.2 Model B changed from CatBoost to Random Forest under ALL6; regenerated Random Forest SHAP with aligned transformed test features.
- `outputs/results/shap_EB68_2_Model_B_RandomForest_importance_ALL6.csv`
- `outputs/figures/shap_EB68_2_Model_B_RandomForest_bar_ALL6.png`
- `outputs/figures/shap_EB68_2_Model_B_RandomForest_dot_ALL6.png`
- EB68.2 Model A remains Random Forest, matching the existing best-model SHAP family; no additional ALL6 SHAP regeneration was required by model selection.

## SVM / ANN Explainability Implications

- No setting selected SVM or ANN as best, so tree-based SHAP remains appropriate for EB68.2 best models when the best models are tree ensembles.

## Sanity-Check Provenance

- The country-only baseline and shuffled-label leakage sanity check were not rerun during this repository-alignment cleanup. The existing values are retained as provenance only and do not affect the ALL6 model ranking or reported ALL6 metric tables.
- The superseded country-only baseline CSV has been archived under `archive_old/old_outputs/outputs_FIXED2/`; the existing leakage sanity-check report remains in `outputs/reports/` for traceability.

## Warnings

- SHAP and feature importance are model-level attribution methods, not causal evidence.
- `country_recycle_rate` is target-encoded aggregate predictive context, not a direct infrastructure or policy measurement.
- The random train/test split evaluates known-country respondent prediction, not unseen-country generalization.
- FEB316 uses oversampling only inside training/CV folds; held-out test rows remain untouched.
- EB68.2 target interpretation must remain tied to `v552` waste-separation mention status.
