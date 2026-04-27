################################################
# KKBox Churn Prediction - Final LogLoss Selection
# Leakage-safe, time-aware, tuned XGBoost champion
################################################

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import pandas as pd
from pathlib import Path

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


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
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob)
    }

    print(f"\n########## {model_name} ##########")
    print("Accuracy:", metrics["accuracy"])
    print("F1:", metrics["f1"])
    print("ROC_AUC:", metrics["roc_auc"])
    print("LogLoss:", metrics["log_loss"])

    return metrics


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
# Prepare Leakage-safe Train/Valid
################################################

def prepare_train_valid_no_logs(train_df, valid_df):
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
# XGBoost Tuning
################################################

def tune_xgboost_for_logloss(X_train, y_train):
    xgb_base = XGBClassifier(
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [150, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 1.5, 2, 3]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=18,
        scoring="neg_log_loss",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
        refit=True
    )

    random_search.fit(X_train, y_train)

    print("\nBest XGBoost Params:")
    print(random_search.best_params_)
    print("Best CV LogLoss:", -random_search.best_score_)

    return random_search.best_estimator_, random_search.best_params_, -random_search.best_score_


################################################
# Main
################################################

def main():
    print("Loading data...")

    project_root = Path(r"D:\GitHub\kkbox-churn-prediction")
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs" / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    train_v2 = pd.read_csv(data_dir / "train_v2.csv")
    transactions_old = pd.read_csv(data_dir / "transactions.csv")
    transactions_v2 = pd.read_csv(data_dir / "transactions_v2.csv")
    members = pd.read_csv(data_dir / "members_v3.csv")

    print("Converting dates...")

    transactions_old["transaction_date"] = pd.to_datetime(
        transactions_old["transaction_date"], format="%Y%m%d"
    )
    transactions_old["membership_expire_date"] = pd.to_datetime(
        transactions_old["membership_expire_date"], format="%Y%m%d"
    )

    transactions_v2["transaction_date"] = pd.to_datetime(
        transactions_v2["transaction_date"], format="%Y%m%d"
    )
    transactions_v2["membership_expire_date"] = pd.to_datetime(
        transactions_v2["membership_expire_date"], format="%Y%m%d"
    )

    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"], format="%Y%m%d"
    )

    print("Building leakage-safe snapshots...")

    snapshot_feb = build_snapshot(train, transactions_old, members, "2017-01-31")
    snapshot_mar = build_snapshot(train_v2, transactions_v2, members, "2017-02-28")

    print("Preparing final train/validation matrices...")

    X_train, X_valid, y_train, y_valid = prepare_train_valid_no_logs(snapshot_feb, snapshot_mar)

    print("X_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)

    print("\nTraining baseline LightGBM...")

    lgbm_model = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=63,
        max_depth=-1
    )
    lgbm_model.fit(X_train, y_train)

    y_pred_lgbm = lgbm_model.predict(X_valid)
    y_prob_lgbm = lgbm_model.predict_proba(X_valid)[:, 1]

    lgbm_metrics = evaluate_classification(
        y_valid, y_pred_lgbm, y_prob_lgbm, "LightGBM Final"
    )

    print("\nTuning XGBoost for LogLoss...")

    xgb_model, best_params, best_cv_logloss = tune_xgboost_for_logloss(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_valid)
    y_prob_xgb = xgb_model.predict_proba(X_valid)[:, 1]

    xgb_metrics = evaluate_classification(
        y_valid, y_pred_xgb, y_prob_xgb, "Tuned XGBoost Final"
    )

    ################################################
    # Save final model
    ################################################

    joblib.dump(xgb_model, output_dir / "kkbox_xgb_final.pkl")

    ################################################
    # Save final results
    ################################################

    final_results = {
        "model_name": "Tuned XGBoost Final",
        "accuracy": xgb_metrics["accuracy"],
        "f1": xgb_metrics["f1"],
        "roc_auc": xgb_metrics["roc_auc"],
        "log_loss": xgb_metrics["log_loss"],
        "best_cv_logloss": float(best_cv_logloss),
        "best_params": best_params
    }

    with open(output_dir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4, default=str)

    print("\nFinal Results Dictionary:")
    print(final_results)

    ################################################
    # Save feature importance
    ################################################

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    print("\nTop 20 Feature Importances:")
    print(feature_importance.head(20))

    ################################################
    # Save Final Figures
    ################################################

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

    figures_dir = project_root / "outputs" / "final_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Feature Importance Figure
    top_n = 20
    fi_plot_df = feature_importance.head(top_n).sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(fi_plot_df["Feature"], fi_plot_df["Importance"])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances - Tuned XGBoost")
    plt.tight_layout()
    plt.savefig(figures_dir / "xgb_feature_importance_top20.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion Matrix Figure
    cm = confusion_matrix(y_valid, y_pred_xgb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix - Tuned XGBoost")
    plt.tight_layout()
    plt.savefig(figures_dir / "xgb_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC Curve Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_valid, y_prob_xgb, ax=ax)
    ax.set_title("ROC Curve - Tuned XGBoost")
    plt.tight_layout()
    plt.savefig(figures_dir / "xgb_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFinal figures saved to:", figures_dir)

    ################################################
    # Save model comparison
    ################################################

    comparison_results = pd.DataFrame([
        {
            "Experiment": "Random Split - LightGBM Tuned",
            "Accuracy": 0.9793039878058828,
            "F1": 0.8874954511099292,
            "ROC_AUC": 0.9932407098515486,
            "LogLoss": 0.05477417331565187,
            "Note": "Optimistic / not leakage-safe"
        },
        {
            "Experiment": "Time-Aware Simple - LightGBM",
            "Accuracy": lgbm_metrics["accuracy"],
            "F1": lgbm_metrics["f1"],
            "ROC_AUC": lgbm_metrics["roc_auc"],
            "LogLoss": lgbm_metrics["log_loss"],
            "Note": "Leakage-safe baseline"
        },
        {
            "Experiment": "Time-Aware Simple - Tuned XGBoost",
            "Accuracy": xgb_metrics["accuracy"],
            "F1": xgb_metrics["f1"],
            "ROC_AUC": xgb_metrics["roc_auc"],
            "LogLoss": xgb_metrics["log_loss"],
            "Note": "Final selected model"
        }
    ])

    comparison_results.to_csv(output_dir / "model_comparison.csv", index=False)

    print("\nModel Comparison:")
    print(comparison_results)

    print("\nDone.")


if __name__ == "__main__":
    main()