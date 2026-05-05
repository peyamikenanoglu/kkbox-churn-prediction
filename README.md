# KKBox Churn Prediction  
## Leakage-Safe, Time-Aware Churn Modeling with Tuned XGBoost

## Overview
This repository presents an end-to-end churn prediction project built on the KKBox dataset, with a strong emphasis on methodological rigor, temporal correctness, and clean project structure.

The goal of this work was not simply to maximize attractive headline metrics. Instead, the project was designed to build a defensible churn prediction pipeline that avoids optimistic evaluation caused by temporal leakage. Early experiments showed that random train/test splits produced unrealistically strong results. The workflow was therefore redesigned around leakage-safe, time-aware validation, and the final model was selected using the competition’s primary metric: **LogLoss**.

The final selected model is a **tuned XGBoost classifier** trained on a leakage-safe February snapshot and evaluated on a March snapshot.

---

## Business Problem
For subscription-based digital platforms, churn prediction is a high-value business problem. If users at risk of leaving can be identified early, the company can intervene through retention campaigns, incentives, or targeted engagement strategies before subscription loss occurs.

In the KKBox setting, the task is to predict whether a user will churn in the following period using historical transaction, subscription, and member-related information.

---

## Project Objective
This project was built to answer four practical questions:

1. Can churn be predicted reliably using transaction and member data?
2. How much do evaluation results change after removing temporal leakage?
3. Which feature groups remain stable under a leakage-safe time-aware setup?
4. Which final model gives the best trustworthy LogLoss?

---

## Dataset
The project uses the KKBox churn competition data, with the following core files:

- `train.csv`
- `train_v2.csv`
- `transactions.csv`
- `transactions_v2.csv`
- `members_v3.csv`

Additional user log data was also explored during research and experimentation. However, user-log-based features were **not retained in the final selected model**, because they did not produce stable improvement under leakage-safe, time-aware evaluation.

---

## Main Challenge: Temporal Leakage
One of the most important findings in this project was that **random split validation was misleading**.

Early experiments using random train/test split produced extremely strong metrics. However, this setup was not reliable for a temporally structured churn problem because future information could leak into training features, directly or indirectly.

To address this, the project was redesigned with a time-aware validation protocol:

- `train.csv` was treated as the **February snapshot**
- `train_v2.csv` was treated as the **March snapshot**
- features were built only from information available **before each cutoff date**
- final validation was performed from **February to March**

This produced lower but much more realistic and defensible performance.

---

## Validation Strategy

| Aspect | Final Approach |
|---|---|
| Evaluation style | Time-aware |
| Leakage handling | Explicitly leakage-safe |
| Training snapshot | February |
| Validation snapshot | March |
| Final selection metric | LogLoss |
| Final selected model | Tuned XGBoost |

---

## Feature Engineering
Feature engineering focused primarily on **transaction** and **member** information.

### Transaction-Based Features
Examples include:

- number of distinct payment methods
- mean and last payment plan days
- mean and last listed price
- mean and last actual amount paid
- mean and last auto-renew behavior
- cancellation statistics
- earliest and latest transaction dates
- latest membership expiry date

### Engineered Features
Additional features were created from raw transaction and member signals:

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

### Features Explored but Not Retained
The following groups were researched during experimentation but excluded from the final selected model:

- user log aggregates
- time-window transaction features
- trend-style user log features
- decline-style user log features

These experiments were useful analytically, but they did not deliver stable improvement under the final leakage-safe time-aware setup.

---

## Models Evaluated

### 1) Optimistic Random-Split Experiment
This experiment produced very strong results, but it was **not leakage-safe** and was therefore rejected as the final methodology.

| Model | Accuracy | F1 | ROC_AUC | LogLoss | Note |
|---|---:|---:|---:|---:|---|
| Tuned LightGBM | 0.979304 | 0.887495 | 0.993241 | 0.054774 | Optimistic, not leakage-safe |

### 2) Trustworthy Leakage-Safe Time-Aware Models
These are the models that matter for final model selection.

| Model | Accuracy | F1 | ROC_AUC | LogLoss | Role |
|---|---:|---:|---:|---:|---|
| LightGBM | 0.811707 | 0.114439 | 0.639168 | 0.382396 | Leakage-safe baseline |
| Tuned XGBoost | 0.809248 | 0.114385 | 0.643268 | 0.377304 | Final selected model |

---

## Final Selected Model
The final selected model is:

### **Tuned XGBoost**
trained on the **leakage-safe February snapshot** and evaluated on the **March snapshot**.

This model was selected because it achieved the **best final LogLoss** among the trustworthy candidate models.

### Best Hyperparameters

| Hyperparameter | Value |
|---|---:|
| `subsample` | 0.8 |
| `reg_lambda` | 1.5 |
| `reg_alpha` | 0.01 |
| `n_estimators` | 200 |
| `min_child_weight` | 3 |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `gamma` | 0 |
| `colsample_bytree` | 0.9 |

---

## Final Results

### Final Selected Model Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.809247548817665 |
| F1 | 0.11438544480837737 |
| ROC_AUC | 0.6432683589855297 |
| LogLoss | 0.37730401356779536 |

### Baseline Comparison

| Comparison | LogLoss |
|---|---:|
| Leakage-safe LightGBM | 0.38239637743883853 |
| Leakage-safe Tuned XGBoost | 0.37730401356779536 |

The tuned XGBoost achieved a small but meaningful improvement in LogLoss and also produced the highest ROC_AUC among the leakage-safe final candidates.

---

## Interpretation of Final Model Behavior

### ROC Behavior
The final ROC_AUC is approximately **0.64**, indicating that the model performs better than random guessing but still operates in a moderate discrimination regime. This is consistent with the fact that the project deliberately removed optimistic leakage and evaluated the model under a realistic temporal shift.

### Confusion Matrix Behavior
The final model correctly identifies a large share of non-churn cases, but still misses many churn cases. In other words, the model is useful but not highly aggressive in catching every churn event. This is also reflected in the relatively low F1 score.

### Feature Importance
The model relies most strongly on features related to subscription continuity, cancellation, and recent payment behavior.

Top signals include:

- `IS_AUTO_RENEW_LAST`
- `IS_CANCEL_LAST`
- `NEW_PRICE_DIFF_LAST`
- `PAYMENT_METHOD_ID_LAST_*`
- `NEW_AUTO_RENEW_RATE`
- `NEW_LAST_TRANS_TO_EXPIRE_DAYS`
- `NEW_NO_MEMBER_INFO`

This is business-consistent: the final model is driven primarily by recent renewal behavior, cancellation status, payment pattern changes, and expiry-related information.

---

## Key Findings

| Finding | Interpretation |
|---|---|
| Random split produced extremely high metrics | Evaluation was overly optimistic |
| Time-aware validation reduced performance | Results became much more realistic |
| User-log-based features were tested | No stable final gain under leakage-safe validation |
| Transaction/member features remained strongest | They formed the final selected model |
| Tuned XGBoost achieved the best final LogLoss | It became the final production candidate |

---

## Why This Repository Does Not Match Kaggle Top Scores
The final metrics in this repository are lower than public top leaderboard-style results, and this is expected.

Likely reasons include:

- top teams used much heavier feature engineering
- top teams exploited user logs more aggressively at scale
- top solutions relied on complex ensembles and stacking
- leaderboard-oriented solutions were often optimized directly for competition scoring behavior
- this repository intentionally prioritized methodological correctness, temporal realism, and leakage-safe evaluation

In other words, this repository is designed as a clean, defensible machine learning project, not a leaderboard-hacking solution.

---

## Repository Structure

<pre><code>kkbox-churn-prediction/
│
├── app/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── outputs/
│   └── final_model/
│       ├── final_results.json
│       ├── feature_importance.csv
│       └── model_comparison.csv
├── src/
│   ├── archive/
│   │   ├── data_loading_and_inspection.py
│   │   └── kkbox_time_aware_research.py
│   ├── experiments/
│   │   └── final_model_evaluation.py
│   └── final_model_logloss_selection.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
</code></pre>

---

## How to Run

Run the final model selection pipeline:

<pre><code>python src/final_model_logloss_selection.py</code></pre>

This script:

- builds leakage-safe February and March snapshots
- prepares final train and validation matrices
- trains a LightGBM baseline
- tunes XGBoost for LogLoss
- evaluates final performance on the March snapshot
- saves the final results and model artifacts

---

## Output Files

| File | Description |
|---|---|
| `outputs/final_model/final_results.json` | Final selected model metrics and best hyperparameters |
| `outputs/final_model/feature_importance.csv` | Feature importance values from the final tuned XGBoost model |
| `outputs/final_model/model_comparison.csv` | Comparison of optimistic and leakage-safe model results |

---

## Reproducibility Notes
This repository intentionally separates:

- **final production-like pipeline** in `src/final_model_logloss_selection.py`
- **research history / exploratory scripts** in `src/archive/`
- **experimental alternative modeling** in `src/experiments/`

This structure keeps the main path of the project clean while preserving research traceability.

---

## Final Note
The strongest contribution of this project is not merely the final XGBoost model. The real contribution is the full workflow:

- identifying optimistic evaluation
- diagnosing temporal leakage risk
- redesigning validation around time-aware snapshots
- testing multiple feature paths
- and selecting a final model using a leakage-safe protocol and the correct competition metric

That is what makes this repository a professional and defensible churn modeling project.

---

## Author
**Peyami Kenanoğlu**
