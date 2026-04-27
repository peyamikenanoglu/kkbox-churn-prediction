################################################
# KKBox Churn Prediction - Final Model Evaluation
# Leakage-safe, time-aware, light ensemble
################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


################################################
# Pandas Settings
################################################

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.show_dimensions", True)


################################################
# Helper Functions
################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def evaluate_classification(y_true, y_pred, y_prob, model_name="Model"):
    print(f"\n########## {model_name} ##########")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print("ROC_AUC:", roc_auc_score(y_true, y_prob))
    print("LogLoss:", log_loss(y_true, y_prob))


def tune_threshold(y_true, y_prob, metric="f1"):
    best_threshold = 0.50
    best_score = -1

    for thr in np.arange(0.10, 0.91, 0.02):
        y_pred = (y_prob >= thr).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred)
        else:
            raise ValueError("Only 'f1' metric is supported for threshold tuning.")

        if score > best_score:
            best_score = score
            best_threshold = thr

    return best_threshold, best_score


################################################
# Transactions Feature Build
################################################

def build_transactions_features(transactions_df, cutoff_date):
    tx = transactions_df[transactions_df["transaction_date"] <= pd.to_datetime(cutoff_date)].copy()
    tx = tx.sort_values(["msno", "transaction_date"])

    tx_agg = tx.groupby("msno").agg({
        "payment_method_id": ["nunique", "last"],
        "payment_plan_days": ["mean", "last"],
        "plan_list_price": ["mean", "last"],
        "actual_amount_paid": ["mean", "last"],
        "is_auto_renew": ["mean", "last"],
        "is_cancel": ["sum", "mean", "last"],
        "transaction_date": ["max", "min"],
        "membership_expire_date": ["max"]
    })

    tx_agg.columns = ["_".join(col).upper() for col in tx_agg.columns]
    tx_agg = tx_agg.reset_index()

    return tx_agg


################################################
# Snapshot Build
################################################

def build_snapshot(labels_df, transactions_df, members_df, cutoff_date):
    df = labels_df.copy()

    tx_agg = build_transactions_features(transactions_df, cutoff_date)

    df = df.merge(tx_agg, how="left", on="msno")
    df = df.merge(members_df, how="left", on="msno")

    # age cleaning
    df["bd"] = df["bd"].astype(float)
    df.loc[(df["bd"] >= 1900) & (df["bd"] <= 2017), "bd"] = 2017 - df["bd"]
    df.loc[(df["bd"] < 10) | (df["bd"] > 100), "bd"] = pd.NA

    # feature engineering
    df["NEW_NO_TRANSACTION"] = df["TRANSACTION_DATE_MAX"].isnull().astype(int)
    df["NEW_NO_MEMBER_INFO"] = df["registration_init_time"].isnull().astype(int)
    df["NEW_GENDER_MISSING"] = df["gender"].isnull().astype(int)

    df["NEW_MEMBERSHIP_DURATION_DAYS"] = (df["MEMBERSHIP_EXPIRE_DATE_MAX"] - df["TRANSACTION_DATE_MIN"]).dt.days
    df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = (df["MEMBERSHIP_EXPIRE_DATE_MAX"] - df["TRANSACTION_DATE_MAX"]).dt.days
    df["NEW_REG_TO_LAST_TRANS_DAYS"] = (df["TRANSACTION_DATE_MAX"] - df["registration_init_time"]).dt.days

    df["NEW_PRICE_DIFF_LAST"] = df["PLAN_LIST_PRICE_LAST"] - df["ACTUAL_AMOUNT_PAID_LAST"]
    df["NEW_PRICE_DIFF_MEAN"] = df["PLAN_LIST_PRICE_MEAN"] - df["ACTUAL_AMOUNT_PAID_MEAN"]

    df["NEW_IS_DISCOUNT_USER"] = (df["ACTUAL_AMOUNT_PAID_MEAN"] < df["PLAN_LIST_PRICE_MEAN"]).astype(float)
    df["NEW_CANCEL_RATE"] = df["IS_CANCEL_MEAN"]
    df["NEW_AUTO_RENEW_RATE"] = df["IS_AUTO_RENEW_MEAN"]

    # fixes
    df.loc[df["NEW_NO_TRANSACTION"] == 1, "NEW_IS_DISCOUNT_USER"] = pd.NA

    df.loc[df["NEW_MEMBERSHIP_DURATION_DAYS"] < 0, "NEW_MEMBERSHIP_DURATION_DAYS"] = pd.NA
    df.loc[df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] < 0, "NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = pd.NA
    df.loc[df["NEW_REG_TO_LAST_TRANS_DAYS"] < 0, "NEW_REG_TO_LAST_TRANS_DAYS"] = pd.NA

    return df


################################################
# Data Prep
################################################

def prepare_train_valid(train_df, valid_df):
    train = train_df.copy()
    valid = valid_df.copy()

    drop_cols = [
        "is_churn",
        "msno",
        "TRANSACTION_DATE_MAX",
        "TRANSACTION_DATE_MIN",
        "MEMBERSHIP_EXPIRE_DATE_MAX",
        "registration_init_time",
        "bd"
    ]

    cat_cols = [
        "PAYMENT_METHOD_ID_LAST",
        "gender",
        "registered_via",
        "city"
    ]

    y_train = train["is_churn"]
    y_valid = valid["is_churn"]

    train = train.drop(drop_cols, axis=1, errors="ignore")
    valid = valid.drop(drop_cols, axis=1, errors="ignore")

    transaction_fill_zero_cols = [
        "PAYMENT_METHOD_ID_NUNIQUE",
        "PAYMENT_PLAN_DAYS_MEAN",
        "PAYMENT_PLAN_DAYS_LAST",
        "PLAN_LIST_PRICE_MEAN",
        "PLAN_LIST_PRICE_LAST",
        "ACTUAL_AMOUNT_PAID_MEAN",
        "ACTUAL_AMOUNT_PAID_LAST",
        "IS_AUTO_RENEW_MEAN",
        "IS_AUTO_RENEW_LAST",
        "IS_CANCEL_SUM",
        "IS_CANCEL_MEAN",
        "IS_CANCEL_LAST",
        "NEW_PRICE_DIFF_LAST",
        "NEW_PRICE_DIFF_MEAN",
        "NEW_IS_DISCOUNT_USER",
        "NEW_CANCEL_RATE",
        "NEW_AUTO_RENEW_RATE"
    ]

    train[transaction_fill_zero_cols] = train[transaction_fill_zero_cols].fillna(0)
    valid[transaction_fill_zero_cols] = valid[transaction_fill_zero_cols].fillna(0)

    train["NEW_MEMBERSHIP_DURATION_DAYS"] = train["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    valid["NEW_MEMBERSHIP_DURATION_DAYS"] = valid["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)

    train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)

    reg_median = train["NEW_REG_TO_LAST_TRANS_DAYS"].median()
    train["NEW_REG_TO_LAST_TRANS_DAYS"] = train["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)
    valid["NEW_REG_TO_LAST_TRANS_DAYS"] = valid["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)

    train = one_hot_encoder(train, cat_cols, drop_first=True)
    valid = one_hot_encoder(valid, cat_cols, drop_first=True)

    valid = valid.reindex(columns=train.columns, fill_value=0)

    return train, valid, y_train, y_valid


################################################
# Main
################################################

def main():
    print("Loading data...")

    train = pd.read_csv("/data/raw/train.csv")
    train_v2 = pd.read_csv("/data/raw/train_v2.csv")
    transactions_old = pd.read_csv("/data/raw/transactions.csv")
    transactions_v2 = pd.read_csv("/data/raw/transactions_v2.csv")
    members = pd.read_csv("/data/raw/members_v3.csv")

    print("Converting date columns...")

    transactions_old["transaction_date"] = pd.to_datetime(transactions_old["transaction_date"], format="%Y%m%d")
    transactions_old["membership_expire_date"] = pd.to_datetime(transactions_old["membership_expire_date"], format="%Y%m%d")

    transactions_v2["transaction_date"] = pd.to_datetime(transactions_v2["transaction_date"], format="%Y%m%d")
    transactions_v2["membership_expire_date"] = pd.to_datetime(transactions_v2["membership_expire_date"], format="%Y%m%d")

    members["registration_init_time"] = pd.to_datetime(members["registration_init_time"], format="%Y%m%d")

    print("Building time-aware snapshots...")

    # leakage-safe snapshots
    snapshot_feb = build_snapshot(train, transactions_old, members, "2017-01-31")
    snapshot_mar = build_snapshot(train_v2, transactions_v2, members, "2017-02-28")

    print("Preparing train/validation data...")

    X_train_full, X_valid_final, y_train_full, y_valid_final = prepare_train_valid(snapshot_feb, snapshot_mar)

    print("Train shape:", X_train_full.shape)
    print("Valid shape:", X_valid_final.shape)

    ################################################
    # Inner split on February only
    # for threshold / ensemble tuning
    ################################################

    X_train_inner, X_hold_inner, y_train_inner, y_hold_inner = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full
    )

    pos_weight = (y_train_inner == 0).sum() / (y_train_inner == 1).sum()

    print("\nTraining LightGBM...")
    lgbm = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        learning_rate=0.05,
        max_depth=-1,
        n_estimators=300,
        num_leaves=63,
        scale_pos_weight=pos_weight
    )
    lgbm.fit(X_train_inner, y_train_inner)
    lgbm_hold_prob = lgbm.predict_proba(X_hold_inner)[:, 1]

    print("Training XGBoost...")
    xgb = XGBClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=pos_weight
    )
    xgb.fit(X_train_inner, y_train_inner)
    xgb_hold_prob = xgb.predict_proba(X_hold_inner)[:, 1]

    # inner single-model metrics
    print("\nInner validation results:")
    evaluate_classification(y_hold_inner, (lgbm_hold_prob >= 0.5).astype(int), lgbm_hold_prob, "LightGBM Inner")
    evaluate_classification(y_hold_inner, (xgb_hold_prob >= 0.5).astype(int), xgb_hold_prob, "XGBoost Inner")

    ################################################
    # Lightweight ensemble search
    ################################################

    best_weight = 0.50
    best_auc = -1

    for w in np.arange(0.0, 1.01, 0.05):
        ens_prob = w * lgbm_hold_prob + (1 - w) * xgb_hold_prob
        auc = roc_auc_score(y_hold_inner, ens_prob)

        if auc > best_auc:
            best_auc = auc
            best_weight = w

    print("\nBest ensemble weight for LightGBM on inner validation:", best_weight)
    print("Best inner ensemble ROC_AUC:", best_auc)

    ens_hold_prob = best_weight * lgbm_hold_prob + (1 - best_weight) * xgb_hold_prob
    best_threshold, best_inner_f1 = tune_threshold(y_hold_inner, ens_hold_prob, metric="f1")

    print("Best threshold on inner validation:", best_threshold)
    print("Best inner F1:", best_inner_f1)

    ################################################
    # Refit on full February and evaluate on March
    ################################################

    pos_weight_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()

    print("\nRefitting models on full February snapshot...")

    lgbm_final = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        learning_rate=0.05,
        max_depth=-1,
        n_estimators=300,
        num_leaves=63,
        scale_pos_weight=pos_weight_full
    )
    lgbm_final.fit(X_train_full, y_train_full)
    lgbm_final_prob = lgbm_final.predict_proba(X_valid_final)[:, 1]

    xgb_final = XGBClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=pos_weight_full
    )
    xgb_final.fit(X_train_full, y_train_full)
    xgb_final_prob = xgb_final.predict_proba(X_valid_final)[:, 1]

    ens_final_prob = best_weight * lgbm_final_prob + (1 - best_weight) * xgb_final_prob

    y_pred_lgbm = (lgbm_final_prob >= 0.5).astype(int)
    y_pred_xgb = (xgb_final_prob >= 0.5).astype(int)
    y_pred_ens_default = (ens_final_prob >= 0.5).astype(int)
    y_pred_ens_tuned = (ens_final_prob >= best_threshold).astype(int)

    print("\nFinal evaluation on March snapshot:")
    evaluate_classification(y_valid_final, y_pred_lgbm, lgbm_final_prob, "LightGBM Final")
    evaluate_classification(y_valid_final, y_pred_xgb, xgb_final_prob, "XGBoost Final")
    evaluate_classification(y_valid_final, y_pred_ens_default, ens_final_prob, "Ensemble Final - Default Threshold")
    evaluate_classification(y_valid_final, y_pred_ens_tuned, ens_final_prob, "Ensemble Final - Tuned Threshold")

    print("\nDone.")


if __name__ == "__main__":
    main()