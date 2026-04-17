# KKBox Churn Prediction

End-to-end churn prediction project based on the KKBox dataset.

## Project Overview
This project aims to predict customer churn using subscription, transaction, and member profile data from the KKBox churn prediction dataset.

## Business Problem
Customer churn prediction helps businesses identify users who are likely to stop renewing their subscriptions. Early prediction can support retention strategies and reduce revenue loss.

## Dataset
Main files planned for the first phase:
- `train_v2.csv`
- `transactions_v2.csv`
- `members_v3.csv`
- `sample_submission_v2.csv`

Dataset source:
- KKBox Churn Prediction Challenge on Kaggle

## Project Goals
- Understand and prepare the dataset
- Build a clean baseline churn model
- Improve performance with feature engineering
- Organize the code into a reusable ML pipeline
- Deploy the final model with FastAPI
- Containerize the app with Docker

## Planned Project Structure
```text
kkbox-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── app/
├── tests/
├── README.md
├── requirements.txt
├── .gitignore
└── config.yaml
