# KKBox Churn Prediction, Leakage-Safe Time-Aware Evaluation

## Project Overview
This project revisits the KKBox churn prediction problem with a strong focus on methodological correctness rather than optimistic leaderboard-style evaluation. The main objective was not only to train a churn classifier, but also to identify and eliminate temporal leakage, redesign validation in a time-aware way, and select the final model using the competition’s primary metric, LogLoss.

The final selected model is a tuned XGBoost classifier trained on a leakage-safe February snapshot and evaluated on a March snapshot. This model was chosen because it achieved the best LogLoss among the leakage-safe, time-aware candidates tested in this project.

## Business Problem
Churn prediction is a high-value problem for subscription-based music streaming services. If users at risk of leaving can be identified early, the company can design retention campaigns, promotions, or targeted engagement strategies before the customer actually churns.

In the KKBox competition setting, the task is to predict whether a subscriber will churn in the following period based on prior subscription, payment, and user behavior signals.

## Dataset
The project uses the KKBox churn competition data, including these core files:

- `train.csv`
- `train_v2.csv`
- `transactions.csv`
- `transactions_v2.csv`
- `members_v3.csv`

Additional user log data was also explored experimentally, but it was not retained in the final selected model because it did not provide stable improvement under leakage-safe time-aware evaluation.

## Main Challenge: Leakage and Validation Strategy
One of the main findings of this project was that random train/test splitting produced highly optimistic results. Early experiments with random split generated extremely strong metrics, but these results were not reliable for a temporally structured churn problem.

To correct this, the project was redesigned with a leakage-safe time-aware validation strategy:

- `train.csv` was treated as the February snapshot
- `train_v2.csv` was treated as the March snapshot
- features were built only from information available before the relevant cutoff date
- final validation was performed from February to March

This change produced much more realistic performance and prevented temporal leakage.

## Feature Engineering
Feature engineering focused primarily on transaction and member data.

### Transaction-based features
Examples include:
- number of distinct payment methods
- mean and last payment plan days
- mean and last listed price
- mean and last actual amount paid
- mean and last auto-renew behavior
- cancellation statistics
- latest and earliest transaction dates
- latest membership expiry date

### Engineered features
Additional features were created from the raw transaction and membership signals, including:
- `NEW_NO_TRANSACTION`
- `NEW_NO_MEMBER_INFO`
- `NEW_GENDER_MISSING`
- `NEW_MEMBERSHIP_DURATION_DAYS`
- `NEW_LAST_TRANS_TO_EXPIRE_DAYS`
- `NEW_REG_TO_LAST_TRANS_DAYS`
- `NEW_PRICE_DIFF_LAST`
- `NEW_PRICE_DIFF_MEAN`
- `NEW_IS_DISCOUNT_USER`
- `NEW_CANCEL_RATE`
- `NEW_AUTO_RENEW_RATE`

### Features explored but not retained
The following were also tested:
- user log aggregates
- time-window transaction features
- trend-style and decline-style user log features

These did not deliver stable improvement under the leakage-safe time-aware setup, so they were excluded from the final selected model.

## Models Evaluated
Several models and evaluation setups were tested during the project.

### Optimistic random-split experiment
This setup produced very strong results, but it was not leakage-safe and was therefore rejected as the final methodology.

- **Random Split, Tuned LightGBM**
  - Accuracy: `0.979304`
  - F1: `0.887495`
  - ROC_AUC: `0.993241`
  - LogLoss: `0.054774`
  - Note: optimistic, not leakage-safe

### Leakage-safe time-aware models
These are the important and trustworthy results.

- **Time-Aware LightGBM**
  - Accuracy: `0.811707`
  - F1: `0.114439`
  - ROC_AUC: `0.639168`
  - LogLoss: `0.382396`

- **Time-Aware Tuned XGBoost**
  - Accuracy: `0.809248`
  - F1: `0.114385`
  - ROC_AUC: `0.643268`
  - LogLoss: `0.377304`

## Final Selected Model
The final selected model is:

**Tuned XGBoost, trained on the leakage-safe February snapshot and evaluated on the March snapshot**

This model was selected because:
- it is leakage-safe
- it is time-aware
- it achieved the best final LogLoss among the trustworthy candidate models

### Best Parameters
- `subsample = 0.8`
- `reg_lambda = 1.5`
- `reg_alpha = 0.01`
- `n_estimators = 200`
- `min_child_weight = 3`
- `max_depth = 8`
- `learning_rate = 0.05`
- `gamma = 0`
- `colsample_bytree = 0.9`

## Final Results
### Final Selected Model Metrics
- **Accuracy:** `0.809247548817665`
- **F1:** `0.11438544480837737`
- **ROC_AUC:** `0.6432683589855297`
- **LogLoss:** `0.37730401356779536`

### Baseline Comparison
- **Leakage-safe LightGBM LogLoss:** `0.38239637743883853`
- **Leakage-safe Tuned XGBoost LogLoss:** `0.37730401356779536`

The tuned XGBoost improved LogLoss slightly and also produced the highest ROC_AUC among the leakage-safe final candidates.

## Feature Importance
The final XGBoost model relied primarily on the following features:

- `IS_AUTO_RENEW_LAST`
- `IS_CANCEL_LAST`
- `NEW_PRICE_DIFF_LAST`
- `PAYMENT_METHOD_ID_LAST_*`
- `NEW_AUTO_RENEW_RATE`
- `NEW_LAST_TRANS_TO_EXPIRE_DAYS`
- `NEW_NO_MEMBER_INFO`

This indicates that the final model was driven mostly by subscription continuity, cancellation behavior, payment patterns, and expiry-related features.

## Key Findings
Several important lessons emerged from this project:

1. Random split produced misleadingly strong metrics for this problem.
2. Leakage-safe, time-aware validation gave lower but much more realistic performance.
3. Transaction and member features were more stable than the user log features explored in this workflow.
4. A tuned XGBoost model provided the best final LogLoss among the trustworthy models.

## Why Results Differ from Kaggle Top Scores
The final results in this repository are much lower than top public Kaggle-style scores, and this is expected.

Possible reasons include:
- top teams likely used much heavier feature engineering
- top solutions used large-scale user log processing more effectively
- many top solutions relied on stacking and multi-model ensembles
- the competition setup may have been optimized directly for leaderboard behavior
- this project intentionally prioritized leakage-safe evaluation and methodological reliability over optimistic leaderboard-like scores

In other words, this repository is designed as a correct and defensible churn modeling project, not a leaderboard-hacking solution.

## Project Structure
```text
kkbox-churn-prediction/
│
├── data/
│   └── raw/
│
├── outputs/
│   └── final_model/
│       ├── kkbox_xgb_final.pkl
│       ├── final_results.json
│       ├── feature_importance.csv
│       └── model_comparison.csv
│
├── final_model_logloss_selection.py
├── final_model_evaluation.py
└── README.md

```

## How to Run
Run the final model selection script:

```bash
python final_model_logloss_selection.py
```

This script:
- builds leakage-safe February and March snapshots
- prepares final train and validation matrices
- trains a LightGBM baseline
- tunes XGBoost for LogLoss
- evaluates final performance on the March snapshot
- saves the final model and result files to `outputs/final_model/`

## Output Files
The final pipeline generates the following artifacts:

- `kkbox_xgb_final.pkl`
- `final_results.json`
- `feature_importance.csv`
- `model_comparison.csv`

## Final Note
The strongest contribution of this project is not just the final model itself, but the full process of identifying optimistic evaluation, redesigning the workflow to avoid temporal leakage, and selecting a final model using a leakage-safe time-aware protocol.

## Author
Peyami Kenanoğlu