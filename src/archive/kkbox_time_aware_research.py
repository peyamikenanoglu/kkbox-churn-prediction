
################################################
# KKBox Churn Prediction - Time-Aware Snapshot Build
################################################

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.show_dimensions', True)


################################################
# Helper Functions
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Nunique #####################")
    print(dataframe.nunique())
def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return na_columns


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

    # bd cleaning
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

    # direct fixes
    df.loc[df["NEW_NO_TRANSACTION"] == 1, "NEW_IS_DISCOUNT_USER"] = pd.NA

    df.loc[df["NEW_MEMBERSHIP_DURATION_DAYS"] < 0, "NEW_MEMBERSHIP_DURATION_DAYS"] = pd.NA
    df.loc[df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] < 0, "NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = pd.NA
    df.loc[df["NEW_REG_TO_LAST_TRANS_DAYS"] < 0, "NEW_REG_TO_LAST_TRANS_DAYS"] = pd.NA

    # transaction-structural missing -> 0
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

    df[transaction_fill_zero_cols] = df[transaction_fill_zero_cols].fillna(0)

    # time feature remaining missing values
    df["NEW_MEMBERSHIP_DURATION_DAYS"] = df["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    df["NEW_REG_TO_LAST_TRANS_DAYS"] = df["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(df["NEW_REG_TO_LAST_TRANS_DAYS"].median())

    return df

################################################
# Data Loading
################################################

train = pd.read_csv("data/raw/train.csv")
train_v2 = pd.read_csv("data/raw/train_v2.csv")
transactions = pd.read_csv("data/raw/transactions_v2.csv")
members = pd.read_csv("data/raw/members_v3.csv")
transactions_old = pd.read_csv("data/raw/transactions.csv")

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], format="%Y%m%d")
transactions["membership_expire_date"] = pd.to_datetime(transactions["membership_expire_date"], format="%Y%m%d")
members["registration_init_time"] = pd.to_datetime(members["registration_init_time"], format="%Y%m%d")
transactions_old["transaction_date"] = pd.to_datetime(transactions_old["transaction_date"], format="%Y%m%d")
transactions_old["membership_expire_date"] = pd.to_datetime(transactions_old["membership_expire_date"], format="%Y%m%d")


################################################
# Snapshot Build
################################################

snapshot_feb = build_snapshot(train, transactions_old, members, "2017-01-31")
snapshot_mar = build_snapshot(train_v2, transactions, members, "2017-02-28")

train.shape, snapshot_feb.shape, snapshot_feb["msno"].is_unique
train_v2.shape, snapshot_mar.shape, snapshot_mar["msno"].is_unique


missing_values_table(snapshot_feb)
missing_values_table(snapshot_mar)
snapshot_feb.head()
snapshot_mar.head()
train.shape, snapshot_feb.shape


train.head()
train_v2.shape
train_v2.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def prepare_train_valid_time_aware(train_df, valid_df):
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

    train = train.drop(drop_cols, axis=1)
    valid = valid.drop(drop_cols, axis=1)

    # structural fill
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

    # train-derived median فقط از فوریه
    reg_median = train["NEW_REG_TO_LAST_TRANS_DAYS"].median()
    train["NEW_REG_TO_LAST_TRANS_DAYS"] = train["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)
    valid["NEW_REG_TO_LAST_TRANS_DAYS"] = valid["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)

    train = one_hot_encoder(train, cat_cols, drop_first=True)
    valid = one_hot_encoder(valid, cat_cols, drop_first=True)

    # column alignment
    valid = valid.reindex(columns=train.columns, fill_value=0)

    return train, valid, y_train, y_valid

X_train_time, X_valid_time, y_train_time, y_valid_time = prepare_train_valid_time_aware(snapshot_feb, snapshot_mar)
X_train_time.shape, X_valid_time.shape, y_train_time.shape, y_valid_time.shape
X_train_time.isnull().sum().sum(), X_valid_time.isnull().sum().sum()


######################################################################
## MODEL
######################################################################

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

lgbm_time = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_time.fit(X_train_time, y_train_time)

y_pred_time = lgbm_time.predict(X_valid_time)
y_prob_time = lgbm_time.predict_proba(X_valid_time)[:, 1]

print("Accuracy:", accuracy_score(y_valid_time, y_pred_time))
print("F1:", f1_score(y_valid_time, y_pred_time))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_time))
print("LogLoss:", log_loss(y_valid_time, y_prob_time))

#####################################################################
## FEATURE ENGINEERING
#####################################################################

def build_window_features(transactions_df, cutoff_date, window_days):
    cutoff_date = pd.to_datetime(cutoff_date)
    start_date = cutoff_date - pd.Timedelta(days=window_days)

    tx = transactions_df[
        (transactions_df["transaction_date"] <= cutoff_date) &
        (transactions_df["transaction_date"] > start_date)
    ].copy()

    tx_window = tx.groupby("msno").agg({
        "payment_method_id": ["nunique"],
        "payment_plan_days": ["mean"],
        "plan_list_price": ["mean"],
        "actual_amount_paid": ["mean"],
        "is_auto_renew": ["mean"],
        "is_cancel": ["sum", "mean"],
        "transaction_date": ["count"]
    })

    tx_window.columns = [
        f"W{window_days}_" + "_".join(col).upper()
        for col in tx_window.columns
    ]

    tx_window = tx_window.reset_index()

    return tx_window
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

    # window-based features
    w15 = build_window_features(transactions_df, cutoff_date, 15)
    w30 = build_window_features(transactions_df, cutoff_date, 30)
    w60 = build_window_features(transactions_df, cutoff_date, 60)
    w90 = build_window_features(transactions_df, cutoff_date, 90)

    tx_agg = tx_agg.merge(w15, how="left", on="msno")
    tx_agg = tx_agg.merge(w30, how="left", on="msno")
    tx_agg = tx_agg.merge(w60, how="left", on="msno")
    tx_agg = tx_agg.merge(w90, how="left", on="msno")

    return tx_agg


snapshot_feb = build_snapshot(train, transactions_old, members, "2017-01-31")
snapshot_mar = build_snapshot(train_v2, transactions, members, "2017-02-28")

snapshot_feb.shape
snapshot_mar.shape
snapshot_feb.columns.tolist()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def prepare_train_valid_time_aware(train_df, valid_df):
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

    train = train.drop(drop_cols, axis=1)
    valid = valid.drop(drop_cols, axis=1)

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
        "NEW_AUTO_RENEW_RATE",

        "W15_PAYMENT_METHOD_ID_NUNIQUE",
        "W15_PAYMENT_PLAN_DAYS_MEAN",
        "W15_PLAN_LIST_PRICE_MEAN",
        "W15_ACTUAL_AMOUNT_PAID_MEAN",
        "W15_IS_AUTO_RENEW_MEAN",
        "W15_IS_CANCEL_SUM",
        "W15_IS_CANCEL_MEAN",
        "W15_TRANSACTION_DATE_COUNT",

        "W30_PAYMENT_METHOD_ID_NUNIQUE",
        "W30_PAYMENT_PLAN_DAYS_MEAN",
        "W30_PLAN_LIST_PRICE_MEAN",
        "W30_ACTUAL_AMOUNT_PAID_MEAN",
        "W30_IS_AUTO_RENEW_MEAN",
        "W30_IS_CANCEL_SUM",
        "W30_IS_CANCEL_MEAN",
        "W30_TRANSACTION_DATE_COUNT",

        "W60_PAYMENT_METHOD_ID_NUNIQUE",
        "W60_PAYMENT_PLAN_DAYS_MEAN",
        "W60_PLAN_LIST_PRICE_MEAN",
        "W60_ACTUAL_AMOUNT_PAID_MEAN",
        "W60_IS_AUTO_RENEW_MEAN",
        "W60_IS_CANCEL_SUM",
        "W60_IS_CANCEL_MEAN",
        "W60_TRANSACTION_DATE_COUNT",

        "W90_PAYMENT_METHOD_ID_NUNIQUE",
        "W90_PAYMENT_PLAN_DAYS_MEAN",
        "W90_PLAN_LIST_PRICE_MEAN",
        "W90_ACTUAL_AMOUNT_PAID_MEAN",
        "W90_IS_AUTO_RENEW_MEAN",
        "W90_IS_CANCEL_SUM",
        "W90_IS_CANCEL_MEAN",
        "W90_TRANSACTION_DATE_COUNT"
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


X_train_time, X_valid_time, y_train_time, y_valid_time = prepare_train_valid_time_aware(snapshot_feb, snapshot_mar)
X_train_time.shape, X_valid_time.shape
X_train_time.isnull().sum().sum(), X_valid_time.isnull().sum().sum()

###########################################################
## MODELING
###########################################################

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

lgbm_time_window = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_time_window.fit(X_train_time, y_train_time)

y_pred_time_window = lgbm_time_window.predict(X_valid_time)
y_prob_time_window = lgbm_time_window.predict_proba(X_valid_time)[:, 1]

print("Accuracy:", accuracy_score(y_valid_time, y_pred_time_window))
print("F1:", f1_score(y_valid_time, y_pred_time_window))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_time_window))
print("LogLoss:", log_loss(y_valid_time, y_prob_time_window))



from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import numpy as np

pos_weight = (y_train_time == 0).sum() / (y_train_time == 1).sum()

lgbm_balanced = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63,
    scale_pos_weight=pos_weight
)

lgbm_balanced.fit(X_train_time, y_train_time)

y_prob_bal = lgbm_balanced.predict_proba(X_valid_time)[:, 1]

best_threshold = 0.50
best_f1 = 0

for thr in np.arange(0.10, 0.91, 0.05):
    y_pred_thr = (y_prob_bal >= thr).astype(int)
    f1 = f1_score(y_valid_time, y_pred_thr)
    print(f"Threshold={thr:.2f} | F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thr

print("\nBest threshold:", best_threshold)
print("Best F1:", best_f1)

y_pred_final = (y_prob_bal >= best_threshold).astype(int)

print("Accuracy:", accuracy_score(y_valid_time, y_pred_final))
print("F1:", f1_score(y_valid_time, y_pred_final))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_bal))
print("LogLoss:", log_loss(y_valid_time, y_prob_bal))


############################################################################
## ADDING USER_LOG AND FEATURE ENGINEERING
############################################################################

user_logs = pd.read_csv("data/raw/user_logs.csv")
user_logs_v2 = pd.read_csv("data/raw/user_logs_v2.csv")

user_logs_sample = pd.read_csv("data/raw/user_logs.csv", nrows=5)
user_logs_sample
user_logs_sample.columns.tolist()

chunk_iter = pd.read_csv("data/raw/user_logs.csv", chunksize=500000)
first_chunk = next(chunk_iter)
first_chunk.head()


def build_user_logs_features(log_path, cutoff_date, chunksize=500000):
    cutoff_date = pd.to_datetime(cutoff_date)

    logs_agg = None

    for chunk in pd.read_csv(log_path, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y%m%d")
        chunk = chunk[chunk["date"] <= cutoff_date].copy()

        if chunk.empty:
            continue

        chunk["total_plays"] = (
            chunk["num_25"] +
            chunk["num_50"] +
            chunk["num_75"] +
            chunk["num_985"] +
            chunk["num_100"]
        )

        chunk_agg = chunk.groupby("msno").agg({
            "date": ["count", "max", "min"],
            "total_secs": ["sum"],
            "num_unq": ["sum"],
            "num_100": ["sum"],
            "total_plays": ["sum"]
        })

        chunk_agg.columns = ["_".join(col).upper() for col in chunk_agg.columns]

        if logs_agg is None:
            logs_agg = chunk_agg.copy()
        else:
            # هم‌راستا کردن indexها
            all_idx = logs_agg.index.union(chunk_agg.index)
            logs_agg = logs_agg.reindex(all_idx)
            chunk_agg = chunk_agg.reindex(all_idx)

            # ستون‌های جمعی
            sum_cols = ["DATE_COUNT", "TOTAL_SECS_SUM", "NUM_UNQ_SUM", "NUM_100_SUM", "TOTAL_PLAYS_SUM"]
            logs_agg[sum_cols] = logs_agg[sum_cols].fillna(0).add(chunk_agg[sum_cols].fillna(0), fill_value=0)

            # ستون‌های تاریخی
            logs_agg["DATE_MAX"] = pd.concat([logs_agg["DATE_MAX"], chunk_agg["DATE_MAX"]], axis=1).max(axis=1)
            logs_agg["DATE_MIN"] = pd.concat([logs_agg["DATE_MIN"], chunk_agg["DATE_MIN"]], axis=1).min(axis=1)

    logs_agg = logs_agg.reset_index()

    return logs_agg

logs_feb = build_user_logs_features("data/raw/user_logs.csv", "2017-01-31")


user_logs_v2.shape

user_logs_v2["date"] = pd.to_datetime(user_logs_v2["date"], format="%Y%m%d")
user_logs_v2["date"].min(), user_logs_v2["date"].max()



import duckdb

con = duckdb.connect()
logs_feb = con.execute("""
    SELECT
        msno,
        COUNT(*) AS LOG_DAYS_COUNT,
        MAX(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MAX,
        MIN(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MIN,
        SUM(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_SUM,
        AVG(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_MEAN,
        SUM(total_secs) AS LOG_TOTAL_SECS_SUM,
        AVG(total_secs) AS LOG_TOTAL_SECS_MEAN,
        SUM(num_unq) AS LOG_NUM_UNQ_SUM,
        AVG(num_unq) AS LOG_NUM_UNQ_MEAN,
        SUM(num_100) AS LOG_NUM_100_SUM,
        AVG(num_100) AS LOG_NUM_100_MEAN
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
    GROUP BY msno
""").df()

logs_feb.shape
logs_feb.head()

logs_mar = con.execute("""
    SELECT
        msno,
        COUNT(*) AS LOG_DAYS_COUNT,
        MAX(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MAX,
        MIN(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MIN,
        SUM(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_SUM,
        AVG(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_MEAN,
        SUM(total_secs) AS LOG_TOTAL_SECS_SUM,
        AVG(total_secs) AS LOG_TOTAL_SECS_MEAN,
        SUM(num_unq) AS LOG_NUM_UNQ_SUM,
        AVG(num_unq) AS LOG_NUM_UNQ_MEAN,
        SUM(num_100) AS LOG_NUM_100_SUM,
        AVG(num_100) AS LOG_NUM_100_MEAN
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-02-28'
    GROUP BY msno
""").df()

logs_mar.shape
logs_mar.head()


def enrich_log_features(logs_df, cutoff_date):
    df = logs_df.copy()
    cutoff_date = pd.to_datetime(cutoff_date)

    df["LOG_AVG_SECS_PER_DAY"] = df["LOG_TOTAL_SECS_SUM"] / df["LOG_DAYS_COUNT"]
    df["LOG_AVG_PLAYS_PER_DAY"] = df["LOG_TOTAL_PLAYS_SUM"] / df["LOG_DAYS_COUNT"]
    df["LOG_AVG_UNQ_PER_DAY"] = df["LOG_NUM_UNQ_SUM"] / df["LOG_DAYS_COUNT"]

    df["LOG_COMPLETION_RATIO"] = df["LOG_NUM_100_SUM"] / df["LOG_TOTAL_PLAYS_SUM"]
    df["LOG_COMPLETION_RATIO"] = df["LOG_COMPLETION_RATIO"].replace([float("inf"), -float("inf")], 0)

    df["LOG_RECENCY_DAYS"] = (cutoff_date - df["LOG_DATE_MAX"]).dt.days

    return df

logs_feb = enrich_log_features(logs_feb, "2017-01-31")
logs_mar = enrich_log_features(logs_mar, "2017-02-28")

snapshot_feb_logs = snapshot_feb.merge(logs_feb, how="left", on="msno")
snapshot_mar_logs = snapshot_mar.merge(logs_mar, how="left", on="msno")

snapshot_feb_logs.shape
snapshot_mar_logs.shape
missing_values_table(snapshot_feb_logs)
missing_values_table(snapshot_mar_logs)





def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def prepare_train_valid_time_aware(train_df, valid_df):
    train = train_df.copy()
    valid = valid_df.copy()

    drop_cols = [
        "LOG_TOTAL_PLAYS_MEAN",
        "LOG_TOTAL_SECS_MEAN",
        "LOG_NUM_UNQ_MEAN",
        "LOG_NUM_100_SUM",
        "LOG_NUM_100_MEAN",
        "LOG_AVG_SECS_PER_DAY",
        "LOG_AVG_PLAYS_PER_DAY",
        "LOG_AVG_UNQ_PER_DAY",
        "LOG_COMPLETION_RATIO",
        "is_churn",
        "msno",
        "TRANSACTION_DATE_MAX",
        "TRANSACTION_DATE_MIN",
        "MEMBERSHIP_EXPIRE_DATE_MAX",
        "registration_init_time",
        "bd",
        "LOG_DATE_MAX",
        "LOG_DATE_MIN",

        "W15_PAYMENT_METHOD_ID_NUNIQUE",
        "W15_PAYMENT_PLAN_DAYS_MEAN",
        "W15_PLAN_LIST_PRICE_MEAN",
        "W15_ACTUAL_AMOUNT_PAID_MEAN",
        "W15_IS_AUTO_RENEW_MEAN",
        "W15_IS_CANCEL_SUM",
        "W15_IS_CANCEL_MEAN",
        "W15_TRANSACTION_DATE_COUNT",

        "W30_PAYMENT_METHOD_ID_NUNIQUE",
        "W30_PAYMENT_PLAN_DAYS_MEAN",
        "W30_PLAN_LIST_PRICE_MEAN",
        "W30_ACTUAL_AMOUNT_PAID_MEAN",
        "W30_IS_AUTO_RENEW_MEAN",
        "W30_IS_CANCEL_SUM",
        "W30_IS_CANCEL_MEAN",
        "W30_TRANSACTION_DATE_COUNT",

        "W60_PAYMENT_METHOD_ID_NUNIQUE",
        "W60_PAYMENT_PLAN_DAYS_MEAN",
        "W60_PLAN_LIST_PRICE_MEAN",
        "W60_ACTUAL_AMOUNT_PAID_MEAN",
        "W60_IS_AUTO_RENEW_MEAN",
        "W60_IS_CANCEL_SUM",
        "W60_IS_CANCEL_MEAN",
        "W60_TRANSACTION_DATE_COUNT",

        "W90_PAYMENT_METHOD_ID_NUNIQUE",
        "W90_PAYMENT_PLAN_DAYS_MEAN",
        "W90_PLAN_LIST_PRICE_MEAN",
        "W90_ACTUAL_AMOUNT_PAID_MEAN",
        "W90_IS_AUTO_RENEW_MEAN",
        "W90_IS_CANCEL_SUM",
        "W90_IS_CANCEL_MEAN",
        "W90_TRANSACTION_DATE_COUNT"
    ]

    cat_cols = [
        "PAYMENT_METHOD_ID_LAST",
        "gender",
        "registered_via",
        "city"
    ]

    y_train = train["is_churn"]
    y_valid = valid["is_churn"]

    train = train.drop(drop_cols, axis=1)
    valid = valid.drop(drop_cols, axis=1)

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

    log_fill_zero_cols = [
        "LOG_DAYS_COUNT",
        "LOG_TOTAL_SECS_SUM",
        "LOG_NUM_UNQ_SUM"
    ]

    train[transaction_fill_zero_cols] = train[transaction_fill_zero_cols].fillna(0)
    valid[transaction_fill_zero_cols] = valid[transaction_fill_zero_cols].fillna(0)

    train[log_fill_zero_cols] = train[log_fill_zero_cols].fillna(0)
    valid[log_fill_zero_cols] = valid[log_fill_zero_cols].fillna(0)

    train["NEW_MEMBERSHIP_DURATION_DAYS"] = train["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    valid["NEW_MEMBERSHIP_DURATION_DAYS"] = valid["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)

    train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)

    reg_median = train["NEW_REG_TO_LAST_TRANS_DAYS"].median()
    train["NEW_REG_TO_LAST_TRANS_DAYS"] = train["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)
    valid["NEW_REG_TO_LAST_TRANS_DAYS"] = valid["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)

    # no-log flag
    train["NEW_NO_LOG_HISTORY"] = (train["LOG_DAYS_COUNT"] == 0).astype(int)
    valid["NEW_NO_LOG_HISTORY"] = (valid["LOG_DAYS_COUNT"] == 0).astype(int)

    # log recency: if no log history, set to train-based max+1
    recency_fill = train["LOG_RECENCY_DAYS"].max(skipna=True) + 1
    train["LOG_RECENCY_DAYS"] = train["LOG_RECENCY_DAYS"].fillna(recency_fill)
    valid["LOG_RECENCY_DAYS"] = valid["LOG_RECENCY_DAYS"].fillna(recency_fill)

    train = one_hot_encoder(train, cat_cols, drop_first=True)
    valid = one_hot_encoder(valid, cat_cols, drop_first=True)

    valid = valid.reindex(columns=train.columns, fill_value=0)

    return train, valid, y_train, y_valid

X_train_time, X_valid_time, y_train_time, y_valid_time = prepare_train_valid_time_aware(snapshot_feb_logs, snapshot_mar_logs)
X_train_time.shape, X_valid_time.shape
X_train_time.isnull().sum().sum(), X_valid_time.isnull().sum().sum()


from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

lgbm_logs = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_logs.fit(X_train_time, y_train_time)

y_pred_logs = lgbm_logs.predict(X_valid_time)
y_prob_logs = lgbm_logs.predict_proba(X_valid_time)[:, 1]

print("Accuracy:", accuracy_score(y_valid_time, y_pred_logs))
print("F1:", f1_score(y_valid_time, y_pred_logs))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_logs))
print("LogLoss:", log_loss(y_valid_time, y_prob_logs))



logs_feb_windows = con.execute("""
WITH base AS (
    SELECT
        msno,
        STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') AS log_date,
        num_25,
        num_50,
        num_75,
        num_985,
        num_100,
        num_unq,
        total_secs,
        (num_25 + num_50 + num_75 + num_985 + num_100) AS total_plays
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
),

agg AS (
    SELECT
        msno,

        MAX(log_date) AS LOG_DATE_MAX,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN 1 ELSE 0 END) AS LOG_DAYS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN 1 ELSE 0 END) AS LOG_DAYS_90,
        SUM(CASE WHEN log_date > DATE '2016-08-04' THEN 1 ELSE 0 END) AS LOG_DAYS_180,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN total_secs ELSE 0 END) AS LOG_SECS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN total_secs ELSE 0 END) AS LOG_SECS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN total_secs ELSE 0 END) AS LOG_SECS_90,
        SUM(CASE WHEN log_date > DATE '2016-08-04' THEN total_secs ELSE 0 END) AS LOG_SECS_180,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN num_unq ELSE 0 END) AS LOG_UNQ_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN num_unq ELSE 0 END) AS LOG_UNQ_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN num_unq ELSE 0 END) AS LOG_UNQ_90,
        SUM(CASE WHEN log_date > DATE '2016-08-04' THEN num_unq ELSE 0 END) AS LOG_UNQ_180,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN num_100 ELSE 0 END) AS LOG_NUM100_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN num_100 ELSE 0 END) AS LOG_NUM100_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN num_100 ELSE 0 END) AS LOG_NUM100_90,
        SUM(CASE WHEN log_date > DATE '2016-08-04' THEN num_100 ELSE 0 END) AS LOG_NUM100_180,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN total_plays ELSE 0 END) AS LOG_PLAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN total_plays ELSE 0 END) AS LOG_PLAYS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN total_plays ELSE 0 END) AS LOG_PLAYS_90,
        SUM(CASE WHEN log_date > DATE '2016-08-04' THEN total_plays ELSE 0 END) AS LOG_PLAYS_180

    FROM base
    GROUP BY msno
)

SELECT * FROM agg
""").df()

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


import pandas as pd
import duckdb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.show_dimensions', True)


train = pd.read_csv("data/raw/train.csv")
train_v2 = pd.read_csv("data/raw/train_v2.csv")

transactions_old = pd.read_csv("data/raw/transactions.csv")
transactions = pd.read_csv("data/raw/transactions_v2.csv")
members = pd.read_csv("data/raw/members_v3.csv")


transactions_old["transaction_date"] = pd.to_datetime(transactions_old["transaction_date"], format="%Y%m%d")
transactions_old["membership_expire_date"] = pd.to_datetime(transactions_old["membership_expire_date"], format="%Y%m%d")

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], format="%Y%m%d")
transactions["membership_expire_date"] = pd.to_datetime(transactions["membership_expire_date"], format="%Y%m%d")

members["registration_init_time"] = pd.to_datetime(members["registration_init_time"], format="%Y%m%d")


def build_window_features(transactions_df, cutoff_date, window_days):
    cutoff_date = pd.to_datetime(cutoff_date)
    start_date = cutoff_date - pd.Timedelta(days=window_days)

    tx = transactions_df[
        (transactions_df["transaction_date"] <= cutoff_date) &
        (transactions_df["transaction_date"] > start_date)
    ].copy()

    tx_window = tx.groupby("msno").agg({
        "payment_method_id": ["nunique"],
        "payment_plan_days": ["mean"],
        "plan_list_price": ["mean"],
        "actual_amount_paid": ["mean"],
        "is_auto_renew": ["mean"],
        "is_cancel": ["sum", "mean"],
        "transaction_date": ["count"]
    })

    tx_window.columns = [
        f"W{window_days}_" + "_".join(col).upper()
        for col in tx_window.columns
    ]

    tx_window = tx_window.reset_index()
    return tx_window
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

    w15 = build_window_features(transactions_df, cutoff_date, 15)
    w30 = build_window_features(transactions_df, cutoff_date, 30)
    w60 = build_window_features(transactions_df, cutoff_date, 60)
    w90 = build_window_features(transactions_df, cutoff_date, 90)

    tx_agg = tx_agg.merge(w15, how="left", on="msno")
    tx_agg = tx_agg.merge(w30, how="left", on="msno")
    tx_agg = tx_agg.merge(w60, how="left", on="msno")
    tx_agg = tx_agg.merge(w90, how="left", on="msno")

    return tx_agg
def build_snapshot(labels_df, transactions_df, members_df, cutoff_date):
    df = labels_df.copy()

    tx_agg = build_transactions_features(transactions_df, cutoff_date)

    df = df.merge(tx_agg, how="left", on="msno")
    df = df.merge(members_df, how="left", on="msno")

    df["bd"] = df["bd"].astype(float)
    df.loc[(df["bd"] >= 1900) & (df["bd"] <= 2017), "bd"] = 2017 - df["bd"]
    df.loc[(df["bd"] < 10) | (df["bd"] > 100), "bd"] = pd.NA

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

    df.loc[df["NEW_NO_TRANSACTION"] == 1, "NEW_IS_DISCOUNT_USER"] = pd.NA

    df.loc[df["NEW_MEMBERSHIP_DURATION_DAYS"] < 0, "NEW_MEMBERSHIP_DURATION_DAYS"] = pd.NA
    df.loc[df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] < 0, "NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = pd.NA
    df.loc[df["NEW_REG_TO_LAST_TRANS_DAYS"] < 0, "NEW_REG_TO_LAST_TRANS_DAYS"] = pd.NA

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

    df[transaction_fill_zero_cols] = df[transaction_fill_zero_cols].fillna(0)
    df["NEW_MEMBERSHIP_DURATION_DAYS"] = df["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    df["NEW_REG_TO_LAST_TRANS_DAYS"] = df["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(df["NEW_REG_TO_LAST_TRANS_DAYS"].median())

    return df
def enrich_log_features(logs_df, cutoff_date):
    df = logs_df.copy()
    cutoff_date = pd.to_datetime(cutoff_date)

    df["LOG_AVG_SECS_PER_DAY"] = df["LOG_TOTAL_SECS_SUM"] / df["LOG_DAYS_COUNT"]
    df["LOG_AVG_PLAYS_PER_DAY"] = df["LOG_TOTAL_PLAYS_SUM"] / df["LOG_DAYS_COUNT"]
    df["LOG_AVG_UNQ_PER_DAY"] = df["LOG_NUM_UNQ_SUM"] / df["LOG_DAYS_COUNT"]

    df["LOG_COMPLETION_RATIO"] = df["LOG_NUM_100_SUM"] / df["LOG_TOTAL_PLAYS_SUM"]
    df["LOG_COMPLETION_RATIO"] = df["LOG_COMPLETION_RATIO"].replace([float("inf"), -float("inf")], 0)

    df["LOG_RECENCY_DAYS"] = (cutoff_date - df["LOG_DATE_MAX"]).dt.days

    return df
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def prepare_train_valid_time_aware(train_df, valid_df):
    train = train_df.copy()
    valid = valid_df.copy()

    drop_cols = [
        "is_churn",
        "msno",
        "TRANSACTION_DATE_MAX",
        "TRANSACTION_DATE_MIN",
        "MEMBERSHIP_EXPIRE_DATE_MAX",
        "registration_init_time",
        "bd",
        "LOG_DATE_MAX",
        "LOG_DATE_MIN",

        "W15_PAYMENT_METHOD_ID_NUNIQUE",
        "W15_PAYMENT_PLAN_DAYS_MEAN",
        "W15_PLAN_LIST_PRICE_MEAN",
        "W15_ACTUAL_AMOUNT_PAID_MEAN",
        "W15_IS_AUTO_RENEW_MEAN",
        "W15_IS_CANCEL_SUM",
        "W15_IS_CANCEL_MEAN",
        "W15_TRANSACTION_DATE_COUNT",

        "W30_PAYMENT_METHOD_ID_NUNIQUE",
        "W30_PAYMENT_PLAN_DAYS_MEAN",
        "W30_PLAN_LIST_PRICE_MEAN",
        "W30_ACTUAL_AMOUNT_PAID_MEAN",
        "W30_IS_AUTO_RENEW_MEAN",
        "W30_IS_CANCEL_SUM",
        "W30_IS_CANCEL_MEAN",
        "W30_TRANSACTION_DATE_COUNT",

        "W60_PAYMENT_METHOD_ID_NUNIQUE",
        "W60_PAYMENT_PLAN_DAYS_MEAN",
        "W60_PLAN_LIST_PRICE_MEAN",
        "W60_ACTUAL_AMOUNT_PAID_MEAN",
        "W60_IS_AUTO_RENEW_MEAN",
        "W60_IS_CANCEL_SUM",
        "W60_IS_CANCEL_MEAN",
        "W60_TRANSACTION_DATE_COUNT",

        "W90_PAYMENT_METHOD_ID_NUNIQUE",
        "W90_PAYMENT_PLAN_DAYS_MEAN",
        "W90_PLAN_LIST_PRICE_MEAN",
        "W90_ACTUAL_AMOUNT_PAID_MEAN",
        "W90_IS_AUTO_RENEW_MEAN",
        "W90_IS_CANCEL_SUM",
        "W90_IS_CANCEL_MEAN",
        "W90_TRANSACTION_DATE_COUNT",

        "LOG_TOTAL_PLAYS_MEAN",
        "LOG_TOTAL_SECS_MEAN",
        "LOG_NUM_UNQ_MEAN",
        "LOG_NUM_100_SUM",
        "LOG_NUM_100_MEAN",
        "LOG_AVG_SECS_PER_DAY",
        "LOG_AVG_PLAYS_PER_DAY",
        "LOG_AVG_UNQ_PER_DAY",
        "LOG_COMPLETION_RATIO",
    ]

    cat_cols = [
        "PAYMENT_METHOD_ID_LAST",
        "gender",
        "registered_via",
        "city"
    ]

    y_train = train["is_churn"]
    y_valid = valid["is_churn"]

    train = train.drop(drop_cols, axis=1)
    valid = valid.drop(drop_cols, axis=1)

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

    log_fill_zero_cols = [
        "LOG_DAYS_COUNT",
        "LOG_TOTAL_SECS_SUM",
        "LOG_NUM_UNQ_SUM"
    ]

    train[transaction_fill_zero_cols] = train[transaction_fill_zero_cols].fillna(0)
    valid[transaction_fill_zero_cols] = valid[transaction_fill_zero_cols].fillna(0)

    train[log_fill_zero_cols] = train[log_fill_zero_cols].fillna(0)
    valid[log_fill_zero_cols] = valid[log_fill_zero_cols].fillna(0)

    train["NEW_MEMBERSHIP_DURATION_DAYS"] = train["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    valid["NEW_MEMBERSHIP_DURATION_DAYS"] = valid["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)

    train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)

    reg_median = train["NEW_REG_TO_LAST_TRANS_DAYS"].median()
    train["NEW_REG_TO_LAST_TRANS_DAYS"] = train["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)
    valid["NEW_REG_TO_LAST_TRANS_DAYS"] = valid["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)

    train["NEW_NO_LOG_HISTORY"] = (train["LOG_DAYS_COUNT"] == 0).astype(int)
    valid["NEW_NO_LOG_HISTORY"] = (valid["LOG_DAYS_COUNT"] == 0).astype(int)

    recency_fill = train["LOG_RECENCY_DAYS"].max(skipna=True) + 1
    train["LOG_RECENCY_DAYS"] = train["LOG_RECENCY_DAYS"].fillna(recency_fill)
    valid["LOG_RECENCY_DAYS"] = valid["LOG_RECENCY_DAYS"].fillna(recency_fill)

    train = one_hot_encoder(train, cat_cols, drop_first=True)
    valid = one_hot_encoder(valid, cat_cols, drop_first=True)

    valid = valid.reindex(columns=train.columns, fill_value=0)

    return train, valid, y_train, y_valid


con = duckdb.connect()

snapshot_feb = build_snapshot(train, transactions_old, members, "2017-01-31")
snapshot_mar = build_snapshot(train_v2, transactions, members, "2017-02-28")


logs_feb = con.execute("""
    SELECT
        msno,
        COUNT(*) AS LOG_DAYS_COUNT,
        MAX(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MAX,
        MIN(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MIN,
        SUM(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_SUM,
        AVG(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_MEAN,
        SUM(total_secs) AS LOG_TOTAL_SECS_SUM,
        AVG(total_secs) AS LOG_TOTAL_SECS_MEAN,
        SUM(num_unq) AS LOG_NUM_UNQ_SUM,
        AVG(num_unq) AS LOG_NUM_UNQ_MEAN,
        SUM(num_100) AS LOG_NUM_100_SUM,
        AVG(num_100) AS LOG_NUM_100_MEAN
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
    GROUP BY msno
""").df()
logs_mar = con.execute("""
    SELECT
        msno,
        COUNT(*) AS LOG_DAYS_COUNT,
        MAX(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MAX,
        MIN(STRPTIME(CAST(date AS VARCHAR), '%Y%m%d')) AS LOG_DATE_MIN,
        SUM(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_SUM,
        AVG(num_25 + num_50 + num_75 + num_985 + num_100) AS LOG_TOTAL_PLAYS_MEAN,
        SUM(total_secs) AS LOG_TOTAL_SECS_SUM,
        AVG(total_secs) AS LOG_TOTAL_SECS_MEAN,
        SUM(num_unq) AS LOG_NUM_UNQ_SUM,
        AVG(num_unq) AS LOG_NUM_UNQ_MEAN,
        SUM(num_100) AS LOG_NUM_100_SUM,
        AVG(num_100) AS LOG_NUM_100_MEAN
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-02-28'
    GROUP BY msno
""").df()


logs_feb = enrich_log_features(logs_feb, "2017-01-31")
logs_mar = enrich_log_features(logs_mar, "2017-02-28")

snapshot_feb_logs = snapshot_feb.merge(logs_feb, how="left", on="msno")
snapshot_mar_logs = snapshot_mar.merge(logs_mar, how="left", on="msno")

X_train_time, X_valid_time, y_train_time, y_valid_time = prepare_train_valid_time_aware(snapshot_feb_logs, snapshot_mar_logs)


lgbm_logs_small = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_logs_small.fit(X_train_time, y_train_time)

y_pred_logs_small = lgbm_logs_small.predict(X_valid_time)
y_prob_logs_small = lgbm_logs_small.predict_proba(X_valid_time)[:, 1]

print("Accuracy:", accuracy_score(y_valid_time, y_pred_logs_small))
print("F1:", f1_score(y_valid_time, y_pred_logs_small))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_logs_small))
print("LogLoss:", log_loss(y_valid_time, y_prob_logs_small))



logs_feb_windows = con.execute("""
WITH base AS (
    SELECT
        msno,
        STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') AS log_date,
        num_unq,
        total_secs,
        (num_25 + num_50 + num_75 + num_985 + num_100) AS total_plays,
        num_100
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
),
agg AS (
    SELECT
        msno,
        MAX(log_date) AS LOG_DATE_MAX,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN 1 ELSE 0 END) AS LOG_DAYS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN total_secs ELSE 0 END) AS LOG_SECS_30,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN total_secs ELSE 0 END) AS LOG_SECS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN num_unq ELSE 0 END) AS LOG_UNQ_30,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN num_unq ELSE 0 END) AS LOG_UNQ_90,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN total_plays ELSE 0 END) AS LOG_PLAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN total_plays ELSE 0 END) AS LOG_PLAYS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN num_100 ELSE 0 END) AS LOG_NUM100_30,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN num_100 ELSE 0 END) AS LOG_NUM100_90
    FROM base
    GROUP BY msno
)
SELECT * FROM agg
""").df()
logs_mar_windows = con.execute("""
WITH base AS (
    SELECT
        msno,
        STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') AS log_date,
        num_unq,
        total_secs,
        (num_25 + num_50 + num_75 + num_985 + num_100) AS total_plays,
        num_100
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-02-28'
),
agg AS (
    SELECT
        msno,
        MAX(log_date) AS LOG_DATE_MAX,

        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN 1 ELSE 0 END) AS LOG_DAYS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN total_secs ELSE 0 END) AS LOG_SECS_30,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN total_secs ELSE 0 END) AS LOG_SECS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN num_unq ELSE 0 END) AS LOG_UNQ_30,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN num_unq ELSE 0 END) AS LOG_UNQ_90,

        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN total_plays ELSE 0 END) AS LOG_PLAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN total_plays ELSE 0 END) AS LOG_PLAYS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN num_100 ELSE 0 END) AS LOG_NUM100_30,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN num_100 ELSE 0 END) AS LOG_NUM100_90
    FROM base
    GROUP BY msno
)
SELECT * FROM agg
""").df()


logs_feb_windows["LOG_RECENCY_DAYS"] = (
    pd.to_datetime("2017-01-31") - logs_feb_windows["LOG_DATE_MAX"]
).dt.days

logs_feb_windows["LOG_ACTIVITY_RATIO_30_90"] = logs_feb_windows["LOG_DAYS_30"] / (logs_feb_windows["LOG_DAYS_90"] + 1)
logs_feb_windows["LOG_SECS_RATIO_30_90"] = logs_feb_windows["LOG_SECS_30"] / (logs_feb_windows["LOG_SECS_90"] + 1)
logs_feb_windows["LOG_UNQ_RATIO_30_90"] = logs_feb_windows["LOG_UNQ_30"] / (logs_feb_windows["LOG_UNQ_90"] + 1)

logs_feb_windows["LOG_SECS_DROP_30_vs_90"] = logs_feb_windows["LOG_SECS_90"] - logs_feb_windows["LOG_SECS_30"]
logs_feb_windows["LOG_DAYS_DROP_30_vs_90"] = logs_feb_windows["LOG_DAYS_90"] - logs_feb_windows["LOG_DAYS_30"]

logs_feb_windows["LOG_COMPLETION_RATIO_30"] = logs_feb_windows["LOG_NUM100_30"] / (logs_feb_windows["LOG_PLAYS_30"] + 1)
logs_feb_windows["LOG_COMPLETION_RATIO_90"] = logs_feb_windows["LOG_NUM100_90"] / (logs_feb_windows["LOG_PLAYS_90"] + 1)



logs_mar_windows["LOG_RECENCY_DAYS"] = (
    pd.to_datetime("2017-02-28") - logs_mar_windows["LOG_DATE_MAX"]
).dt.days

logs_mar_windows["LOG_ACTIVITY_RATIO_30_90"] = logs_mar_windows["LOG_DAYS_30"] / (logs_mar_windows["LOG_DAYS_90"] + 1)
logs_mar_windows["LOG_SECS_RATIO_30_90"] = logs_mar_windows["LOG_SECS_30"] / (logs_mar_windows["LOG_SECS_90"] + 1)
logs_mar_windows["LOG_UNQ_RATIO_30_90"] = logs_mar_windows["LOG_UNQ_30"] / (logs_mar_windows["LOG_UNQ_90"] + 1)

logs_mar_windows["LOG_SECS_DROP_30_vs_90"] = logs_mar_windows["LOG_SECS_90"] - logs_mar_windows["LOG_SECS_30"]
logs_mar_windows["LOG_DAYS_DROP_30_vs_90"] = logs_mar_windows["LOG_DAYS_90"] - logs_mar_windows["LOG_DAYS_30"]

logs_mar_windows["LOG_COMPLETION_RATIO_30"] = logs_mar_windows["LOG_NUM100_30"] / (logs_mar_windows["LOG_PLAYS_30"] + 1)
logs_mar_windows["LOG_COMPLETION_RATIO_90"] = logs_mar_windows["LOG_NUM100_90"] / (logs_mar_windows["LOG_PLAYS_90"] + 1)



logs_feb_windows.shape
logs_mar_windows.shape
logs_feb_windows.head()
logs_mar_windows.head()



logs_feb_trend = logs_feb_windows[[
    "msno",
    "LOG_RECENCY_DAYS",
    "LOG_ACTIVITY_RATIO_30_90",
    "LOG_SECS_RATIO_30_90",
    "LOG_UNQ_RATIO_30_90",
    "LOG_SECS_DROP_30_vs_90",
    "LOG_DAYS_DROP_30_vs_90",
    "LOG_COMPLETION_RATIO_30",
    "LOG_COMPLETION_RATIO_90"
]].copy()
logs_mar_trend = logs_mar_windows[[
    "msno",
    "LOG_RECENCY_DAYS",
    "LOG_ACTIVITY_RATIO_30_90",
    "LOG_SECS_RATIO_30_90",
    "LOG_UNQ_RATIO_30_90",
    "LOG_SECS_DROP_30_vs_90",
    "LOG_DAYS_DROP_30_vs_90",
    "LOG_COMPLETION_RATIO_30",
    "LOG_COMPLETION_RATIO_90"
]].copy()

snapshot_feb_trend = snapshot_feb.merge(logs_feb_trend, how="left", on="msno")
snapshot_mar_trend = snapshot_mar.merge(logs_mar_trend, how="left", on="msno")

snapshot_feb_trend.shape
snapshot_mar_trend.shape


def prepare_train_valid_time_aware_trend(train_df, valid_df):
    train = train_df.copy()
    valid = valid_df.copy()

    drop_cols = [
        "is_churn",
        "msno",
        "TRANSACTION_DATE_MAX",
        "TRANSACTION_DATE_MIN",
        "MEMBERSHIP_EXPIRE_DATE_MAX",
        "registration_init_time",
        "bd",

        "W15_PAYMENT_METHOD_ID_NUNIQUE",
        "W15_PAYMENT_PLAN_DAYS_MEAN",
        "W15_PLAN_LIST_PRICE_MEAN",
        "W15_ACTUAL_AMOUNT_PAID_MEAN",
        "W15_IS_AUTO_RENEW_MEAN",
        "W15_IS_CANCEL_SUM",
        "W15_IS_CANCEL_MEAN",
        "W15_TRANSACTION_DATE_COUNT",

        "W30_PAYMENT_METHOD_ID_NUNIQUE",
        "W30_PAYMENT_PLAN_DAYS_MEAN",
        "W30_PLAN_LIST_PRICE_MEAN",
        "W30_ACTUAL_AMOUNT_PAID_MEAN",
        "W30_IS_AUTO_RENEW_MEAN",
        "W30_IS_CANCEL_SUM",
        "W30_IS_CANCEL_MEAN",
        "W30_TRANSACTION_DATE_COUNT",

        "W60_PAYMENT_METHOD_ID_NUNIQUE",
        "W60_PAYMENT_PLAN_DAYS_MEAN",
        "W60_PLAN_LIST_PRICE_MEAN",
        "W60_ACTUAL_AMOUNT_PAID_MEAN",
        "W60_IS_AUTO_RENEW_MEAN",
        "W60_IS_CANCEL_SUM",
        "W60_IS_CANCEL_MEAN",
        "W60_TRANSACTION_DATE_COUNT",

        "W90_PAYMENT_METHOD_ID_NUNIQUE",
        "W90_PAYMENT_PLAN_DAYS_MEAN",
        "W90_PLAN_LIST_PRICE_MEAN",
        "W90_ACTUAL_AMOUNT_PAID_MEAN",
        "W90_IS_AUTO_RENEW_MEAN",
        "W90_IS_CANCEL_SUM",
        "W90_IS_CANCEL_MEAN",
        "W90_TRANSACTION_DATE_COUNT"
    ]

    cat_cols = [
        "PAYMENT_METHOD_ID_LAST",
        "gender",
        "registered_via",
        "city"
    ]

    y_train = train["is_churn"]
    y_valid = valid["is_churn"]

    train = train.drop(drop_cols, axis=1)
    valid = valid.drop(drop_cols, axis=1)

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

    trend_fill_zero_cols = [
        "LOG_ACTIVITY_RATIO_30_90",
        "LOG_SECS_RATIO_30_90",
        "LOG_UNQ_RATIO_30_90",
        "LOG_SECS_DROP_30_vs_90",
        "LOG_DAYS_DROP_30_vs_90",
        "LOG_COMPLETION_RATIO_30",
        "LOG_COMPLETION_RATIO_90"
    ]

    train[transaction_fill_zero_cols] = train[transaction_fill_zero_cols].fillna(0)
    valid[transaction_fill_zero_cols] = valid[transaction_fill_zero_cols].fillna(0)

    train[trend_fill_zero_cols] = train[trend_fill_zero_cols].fillna(0)
    valid[trend_fill_zero_cols] = valid[trend_fill_zero_cols].fillna(0)

    train["NEW_MEMBERSHIP_DURATION_DAYS"] = train["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    valid["NEW_MEMBERSHIP_DURATION_DAYS"] = valid["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)

    train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = train["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = valid["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)

    reg_median = train["NEW_REG_TO_LAST_TRANS_DAYS"].median()
    train["NEW_REG_TO_LAST_TRANS_DAYS"] = train["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)
    valid["NEW_REG_TO_LAST_TRANS_DAYS"] = valid["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(reg_median)

    recency_fill = train["LOG_RECENCY_DAYS"].max(skipna=True) + 1
    train["LOG_RECENCY_DAYS"] = train["LOG_RECENCY_DAYS"].fillna(recency_fill)
    valid["LOG_RECENCY_DAYS"] = valid["LOG_RECENCY_DAYS"].fillna(recency_fill)

    train["NEW_NO_LOG_TREND_HISTORY"] = train["LOG_RECENCY_DAYS"].isnull().astype(int)
    valid["NEW_NO_LOG_TREND_HISTORY"] = valid["LOG_RECENCY_DAYS"].isnull().astype(int)

    train = one_hot_encoder(train, cat_cols, drop_first=True)
    valid = one_hot_encoder(valid, cat_cols, drop_first=True)

    valid = valid.reindex(columns=train.columns, fill_value=0)

    return train, valid, y_train, y_valid

X_train_time, X_valid_time, y_train_time, y_valid_time = prepare_train_valid_time_aware_trend(snapshot_feb_trend, snapshot_mar_trend)

lgbm_trend = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_trend.fit(X_train_time, y_train_time)

y_pred_trend = lgbm_trend.predict(X_valid_time)
y_prob_trend = lgbm_trend.predict_proba(X_valid_time)[:, 1]

print("Accuracy:", accuracy_score(y_valid_time, y_pred_trend))
print("F1:", f1_score(y_valid_time, y_pred_trend))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_trend))
print("LogLoss:", log_loss(y_valid_time, y_prob_trend))




logs_feb_decline = con.execute("""
WITH base AS (
    SELECT
        msno,
        STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') AS log_date,
        num_unq,
        total_secs
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
),
agg AS (
    SELECT
        msno,
        MAX(log_date) AS LOG_DATE_MAX,

        SUM(CASE WHEN log_date > DATE '2017-01-21' THEN 1 ELSE 0 END) AS LOG_DAYS_10,
        SUM(CASE WHEN log_date > DATE '2017-01-11' THEN 1 ELSE 0 END) AS LOG_DAYS_20,
        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN 1 ELSE 0 END) AS LOG_DAYS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN 1 ELSE 0 END) AS LOG_DAYS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-21' THEN total_secs ELSE 0 END) AS LOG_SECS_10,
        SUM(CASE WHEN log_date > DATE '2017-01-11' THEN total_secs ELSE 0 END) AS LOG_SECS_20,
        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN total_secs ELSE 0 END) AS LOG_SECS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN total_secs ELSE 0 END) AS LOG_SECS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN total_secs ELSE 0 END) AS LOG_SECS_90,

        SUM(CASE WHEN log_date > DATE '2017-01-21' THEN num_unq ELSE 0 END) AS LOG_UNQ_10,
        SUM(CASE WHEN log_date > DATE '2017-01-11' THEN num_unq ELSE 0 END) AS LOG_UNQ_20,
        SUM(CASE WHEN log_date > DATE '2017-01-01' THEN num_unq ELSE 0 END) AS LOG_UNQ_30,
        SUM(CASE WHEN log_date > DATE '2016-12-02' THEN num_unq ELSE 0 END) AS LOG_UNQ_60,
        SUM(CASE WHEN log_date > DATE '2016-11-02' THEN num_unq ELSE 0 END) AS LOG_UNQ_90
    FROM base
    GROUP BY msno
)
SELECT * FROM agg
""").df()
logs_mar_decline = con.execute("""
WITH base AS (
    SELECT
        msno,
        STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') AS log_date,
        num_unq,
        total_secs
    FROM read_csv_auto('data/raw/user_logs.csv')
    WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-02-28'
),
agg AS (
    SELECT
        msno,
        MAX(log_date) AS LOG_DATE_MAX,

        SUM(CASE WHEN log_date > DATE '2017-02-18' THEN 1 ELSE 0 END) AS LOG_DAYS_10,
        SUM(CASE WHEN log_date > DATE '2017-02-08' THEN 1 ELSE 0 END) AS LOG_DAYS_20,
        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-30' THEN 1 ELSE 0 END) AS LOG_DAYS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN 1 ELSE 0 END) AS LOG_DAYS_90,

        SUM(CASE WHEN log_date > DATE '2017-02-18' THEN total_secs ELSE 0 END) AS LOG_SECS_10,
        SUM(CASE WHEN log_date > DATE '2017-02-08' THEN total_secs ELSE 0 END) AS LOG_SECS_20,
        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN total_secs ELSE 0 END) AS LOG_SECS_30,
        SUM(CASE WHEN log_date > DATE '2016-12-30' THEN total_secs ELSE 0 END) AS LOG_SECS_60,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN total_secs ELSE 0 END) AS LOG_SECS_90,

        SUM(CASE WHEN log_date > DATE '2017-02-18' THEN num_unq ELSE 0 END) AS LOG_UNQ_10,
        SUM(CASE WHEN log_date > DATE '2017-02-08' THEN num_unq ELSE 0 END) AS LOG_UNQ_20,
        SUM(CASE WHEN log_date > DATE '2017-01-29' THEN num_unq ELSE 0 END) AS LOG_UNQ_30,
        SUM(CASE WHEN log_date > DATE '2016-12-30' THEN num_unq ELSE 0 END) AS LOG_UNQ_60,
        SUM(CASE WHEN log_date > DATE '2016-11-30' THEN num_unq ELSE 0 END) AS LOG_UNQ_90
    FROM base
    GROUP BY msno
)
SELECT * FROM agg
""").df()


logs_feb_decline["LOG_RECENCY_DAYS"] = (
    pd.to_datetime("2017-01-31") - logs_feb_decline["LOG_DATE_MAX"]
).dt.days

logs_feb_decline["LOG_SECS_RATIO_10_30"] = logs_feb_decline["LOG_SECS_10"] / (logs_feb_decline["LOG_SECS_30"] + 1)
logs_feb_decline["LOG_SECS_RATIO_10_60"] = logs_feb_decline["LOG_SECS_10"] / (logs_feb_decline["LOG_SECS_60"] + 1)
logs_feb_decline["LOG_SECS_RATIO_10_90"] = logs_feb_decline["LOG_SECS_10"] / (logs_feb_decline["LOG_SECS_90"] + 1)
logs_feb_decline["LOG_SECS_RATIO_20_60"] = logs_feb_decline["LOG_SECS_20"] / (logs_feb_decline["LOG_SECS_60"] + 1)
logs_feb_decline["LOG_SECS_RATIO_30_90"] = logs_feb_decline["LOG_SECS_30"] / (logs_feb_decline["LOG_SECS_90"] + 1)

logs_feb_decline["LOG_UNQ_RATIO_10_30"] = logs_feb_decline["LOG_UNQ_10"] / (logs_feb_decline["LOG_UNQ_30"] + 1)
logs_feb_decline["LOG_UNQ_RATIO_10_60"] = logs_feb_decline["LOG_UNQ_10"] / (logs_feb_decline["LOG_UNQ_60"] + 1)
logs_feb_decline["LOG_UNQ_RATIO_30_90"] = logs_feb_decline["LOG_UNQ_30"] / (logs_feb_decline["LOG_UNQ_90"] + 1)

logs_feb_decline["LOG_DAYS_RATIO_10_30"] = logs_feb_decline["LOG_DAYS_10"] / (logs_feb_decline["LOG_DAYS_30"] + 1)
logs_feb_decline["LOG_DAYS_RATIO_10_60"] = logs_feb_decline["LOG_DAYS_10"] / (logs_feb_decline["LOG_DAYS_60"] + 1)
logs_feb_decline["LOG_DAYS_RATIO_30_90"] = logs_feb_decline["LOG_DAYS_30"] / (logs_feb_decline["LOG_DAYS_90"] + 1)

logs_feb_decline["LOG_SECS_DROP_10_90"] = logs_feb_decline["LOG_SECS_90"] - logs_feb_decline["LOG_SECS_10"]
logs_feb_decline["LOG_UNQ_DROP_10_90"] = logs_feb_decline["LOG_UNQ_90"] - logs_feb_decline["LOG_UNQ_10"]
logs_feb_decline["LOG_DAYS_DROP_10_90"] = logs_feb_decline["LOG_DAYS_90"] - logs_feb_decline["LOG_DAYS_10"]



logs_mar_decline["LOG_RECENCY_DAYS"] = (
    pd.to_datetime("2017-02-28") - logs_mar_decline["LOG_DATE_MAX"]
).dt.days

logs_mar_decline["LOG_SECS_RATIO_10_30"] = logs_mar_decline["LOG_SECS_10"] / (logs_mar_decline["LOG_SECS_30"] + 1)
logs_mar_decline["LOG_SECS_RATIO_10_60"] = logs_mar_decline["LOG_SECS_10"] / (logs_mar_decline["LOG_SECS_60"] + 1)
logs_mar_decline["LOG_SECS_RATIO_10_90"] = logs_mar_decline["LOG_SECS_10"] / (logs_mar_decline["LOG_SECS_90"] + 1)
logs_mar_decline["LOG_SECS_RATIO_20_60"] = logs_mar_decline["LOG_SECS_20"] / (logs_mar_decline["LOG_SECS_60"] + 1)
logs_mar_decline["LOG_SECS_RATIO_30_90"] = logs_mar_decline["LOG_SECS_30"] / (logs_mar_decline["LOG_SECS_90"] + 1)

logs_mar_decline["LOG_UNQ_RATIO_10_30"] = logs_mar_decline["LOG_UNQ_10"] / (logs_mar_decline["LOG_UNQ_30"] + 1)
logs_mar_decline["LOG_UNQ_RATIO_10_60"] = logs_mar_decline["LOG_UNQ_10"] / (logs_mar_decline["LOG_UNQ_60"] + 1)
logs_mar_decline["LOG_UNQ_RATIO_30_90"] = logs_mar_decline["LOG_UNQ_30"] / (logs_mar_decline["LOG_UNQ_90"] + 1)

logs_mar_decline["LOG_DAYS_RATIO_10_30"] = logs_mar_decline["LOG_DAYS_10"] / (logs_mar_decline["LOG_DAYS_30"] + 1)
logs_mar_decline["LOG_DAYS_RATIO_10_60"] = logs_mar_decline["LOG_DAYS_10"] / (logs_mar_decline["LOG_DAYS_60"] + 1)
logs_mar_decline["LOG_DAYS_RATIO_30_90"] = logs_mar_decline["LOG_DAYS_30"] / (logs_mar_decline["LOG_DAYS_90"] + 1)

logs_mar_decline["LOG_SECS_DROP_10_90"] = logs_mar_decline["LOG_SECS_90"] - logs_mar_decline["LOG_SECS_10"]
logs_mar_decline["LOG_UNQ_DROP_10_90"] = logs_mar_decline["LOG_UNQ_90"] - logs_mar_decline["LOG_UNQ_10"]
logs_mar_decline["LOG_DAYS_DROP_10_90"] = logs_mar_decline["LOG_DAYS_90"] - logs_mar_decline["LOG_DAYS_10"]


logs_feb_decline.shape
logs_mar_decline.shape
logs_feb_decline.head()
logs_mar_decline.head()


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################


import pandas as pd
import duckdb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.show_dimensions', True)


################################################
# Load Data
################################################

train = pd.read_csv("data/raw/train.csv")
train_v2 = pd.read_csv("data/raw/train_v2.csv")

transactions_old = pd.read_csv("data/raw/transactions.csv")
transactions = pd.read_csv("data/raw/transactions_v2.csv")
members = pd.read_csv("data/raw/members_v3.csv")

transactions_old["transaction_date"] = pd.to_datetime(transactions_old["transaction_date"], format="%Y%m%d")
transactions_old["membership_expire_date"] = pd.to_datetime(transactions_old["membership_expire_date"], format="%Y%m%d")

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], format="%Y%m%d")
transactions["membership_expire_date"] = pd.to_datetime(transactions["membership_expire_date"], format="%Y%m%d")

members["registration_init_time"] = pd.to_datetime(members["registration_init_time"], format="%Y%m%d")


################################################
# Helper Functions
################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
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
def build_snapshot(labels_df, transactions_df, members_df, cutoff_date):
    df = labels_df.copy()

    tx_agg = build_transactions_features(transactions_df, cutoff_date)

    df = df.merge(tx_agg, how="left", on="msno")
    df = df.merge(members_df, how="left", on="msno")

    df["bd"] = df["bd"].astype(float)
    df.loc[(df["bd"] >= 1900) & (df["bd"] <= 2017), "bd"] = 2017 - df["bd"]
    df.loc[(df["bd"] < 10) | (df["bd"] > 100), "bd"] = pd.NA

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

    df.loc[df["NEW_NO_TRANSACTION"] == 1, "NEW_IS_DISCOUNT_USER"] = pd.NA

    df.loc[df["NEW_MEMBERSHIP_DURATION_DAYS"] < 0, "NEW_MEMBERSHIP_DURATION_DAYS"] = pd.NA
    df.loc[df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] < 0, "NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = pd.NA
    df.loc[df["NEW_REG_TO_LAST_TRANS_DAYS"] < 0, "NEW_REG_TO_LAST_TRANS_DAYS"] = pd.NA

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

    df[transaction_fill_zero_cols] = df[transaction_fill_zero_cols].fillna(0)
    df["NEW_MEMBERSHIP_DURATION_DAYS"] = df["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
    df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
    df["NEW_REG_TO_LAST_TRANS_DAYS"] = df["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(df["NEW_REG_TO_LAST_TRANS_DAYS"].median())

    return df
def prepare_train_valid_simple_logs(train_df, valid_df):
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

    train = train.drop(drop_cols, axis=1)
    valid = valid.drop(drop_cols, axis=1)

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
        "NEW_AUTO_RENEW_RATE",
        "LOG_DAYS_30",
        "HAS_LOG_LAST_30"
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

    recency_fill = train["LOG_RECENCY_DAYS"].max(skipna=True) + 1
    train["LOG_RECENCY_DAYS"] = train["LOG_RECENCY_DAYS"].fillna(recency_fill)
    valid["LOG_RECENCY_DAYS"] = valid["LOG_RECENCY_DAYS"].fillna(recency_fill)

    train = one_hot_encoder(train, cat_cols, drop_first=True)
    valid = one_hot_encoder(valid, cat_cols, drop_first=True)

    valid = valid.reindex(columns=train.columns, fill_value=0)

    return train, valid, y_train, y_valid

################################################
# Build leakage-safe snapshots
################################################

snapshot_feb = build_snapshot(train, transactions_old, members, "2017-01-31")
snapshot_mar = build_snapshot(train_v2, transactions, members, "2017-02-28")

################################################
# Build only 3 simple log features with DuckDB
################################################

con = duckdb.connect()

logs_feb_simple = con.execute("""
    SELECT
        msno,
        SUM(CASE WHEN STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') > DATE '2017-01-01' 
                 AND STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
                 THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        MAX(CASE WHEN STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-01-31'
                 THEN STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') END) AS LOG_DATE_MAX
    FROM read_csv_auto('data/raw/user_logs.csv')
    GROUP BY msno
""").df()

logs_mar_simple = con.execute("""
    SELECT
        msno,
        SUM(CASE WHEN STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') > DATE '2017-01-29' 
                 AND STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-02-28'
                 THEN 1 ELSE 0 END) AS LOG_DAYS_30,
        MAX(CASE WHEN STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') <= DATE '2017-02-28'
                 THEN STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') END) AS LOG_DATE_MAX
    FROM read_csv_auto('data/raw/user_logs.csv')
    GROUP BY msno
""").df()


################################################
# Create simple log features
################################################

logs_feb_simple["HAS_LOG_LAST_30"] = (logs_feb_simple["LOG_DAYS_30"] > 0).astype(int)
logs_mar_simple["HAS_LOG_LAST_30"] = (logs_mar_simple["LOG_DAYS_30"] > 0).astype(int)

logs_feb_simple["LOG_RECENCY_DAYS"] = (
    pd.to_datetime("2017-01-31") - logs_feb_simple["LOG_DATE_MAX"]
).dt.days

logs_mar_simple["LOG_RECENCY_DAYS"] = (
    pd.to_datetime("2017-02-28") - logs_mar_simple["LOG_DATE_MAX"]
).dt.days

logs_feb_simple = logs_feb_simple[["msno", "LOG_DAYS_30", "HAS_LOG_LAST_30", "LOG_RECENCY_DAYS"]].copy()
logs_mar_simple = logs_mar_simple[["msno", "LOG_DAYS_30", "HAS_LOG_LAST_30", "LOG_RECENCY_DAYS"]].copy()


################################################
# Merge simple logs
################################################

snapshot_feb_simple_logs = snapshot_feb.merge(logs_feb_simple, how="left", on="msno")
snapshot_mar_simple_logs = snapshot_mar.merge(logs_mar_simple, how="left", on="msno")


################################################
# Sample for faster testing
################################################

sample_n = 100000
sample_idx_train = snapshot_feb_simple_logs.sample(n=sample_n, random_state=42).index
sample_idx_valid = snapshot_mar_simple_logs.sample(n=sample_n, random_state=42).index

snapshot_feb_sample = snapshot_feb_simple_logs.loc[sample_idx_train].copy()
snapshot_mar_sample = snapshot_mar_simple_logs.loc[sample_idx_valid].copy()


################################################
# Prepare train/valid
################################################

X_train_time, X_valid_time, y_train_time, y_valid_time = prepare_train_valid_simple_logs(
    snapshot_feb_sample,
    snapshot_mar_sample
)


################################################
# Train LightGBM
################################################

lgbm_simple_logs = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_simple_logs.fit(X_train_time, y_train_time)

y_pred_simple_logs = lgbm_simple_logs.predict(X_valid_time)
y_prob_simple_logs = lgbm_simple_logs.predict_proba(X_valid_time)[:, 1]

print("X_train_time.shape:", X_train_time.shape)
print("X_valid_time.shape:", X_valid_time.shape)
print("Accuracy:", accuracy_score(y_valid_time, y_pred_simple_logs))
print("F1:", f1_score(y_valid_time, y_pred_simple_logs))
print("ROC_AUC:", roc_auc_score(y_valid_time, y_prob_simple_logs))
print("LogLoss:", log_loss(y_valid_time, y_prob_simple_logs))




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
        "bd",

        "W15_PAYMENT_METHOD_ID_NUNIQUE",
        "W15_PAYMENT_PLAN_DAYS_MEAN",
        "W15_PLAN_LIST_PRICE_MEAN",
        "W15_ACTUAL_AMOUNT_PAID_MEAN",
        "W15_IS_AUTO_RENEW_MEAN",
        "W15_IS_CANCEL_SUM",
        "W15_IS_CANCEL_MEAN",
        "W15_TRANSACTION_DATE_COUNT",

        "W30_PAYMENT_METHOD_ID_NUNIQUE",
        "W30_PAYMENT_PLAN_DAYS_MEAN",
        "W30_PLAN_LIST_PRICE_MEAN",
        "W30_ACTUAL_AMOUNT_PAID_MEAN",
        "W30_IS_AUTO_RENEW_MEAN",
        "W30_IS_CANCEL_SUM",
        "W30_IS_CANCEL_MEAN",
        "W30_TRANSACTION_DATE_COUNT",

        "W60_PAYMENT_METHOD_ID_NUNIQUE",
        "W60_PAYMENT_PLAN_DAYS_MEAN",
        "W60_PLAN_LIST_PRICE_MEAN",
        "W60_ACTUAL_AMOUNT_PAID_MEAN",
        "W60_IS_AUTO_RENEW_MEAN",
        "W60_IS_CANCEL_SUM",
        "W60_IS_CANCEL_MEAN",
        "W60_TRANSACTION_DATE_COUNT",

        "W90_PAYMENT_METHOD_ID_NUNIQUE",
        "W90_PAYMENT_PLAN_DAYS_MEAN",
        "W90_PLAN_LIST_PRICE_MEAN",
        "W90_ACTUAL_AMOUNT_PAID_MEAN",
        "W90_IS_AUTO_RENEW_MEAN",
        "W90_IS_CANCEL_SUM",
        "W90_IS_CANCEL_MEAN",
        "W90_TRANSACTION_DATE_COUNT"
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

snapshot_feb_sample_nolog = snapshot_feb.loc[sample_idx_train].copy()
snapshot_mar_sample_nolog = snapshot_mar.loc[sample_idx_valid].copy()

X_train_nolog, X_valid_nolog, y_train_nolog, y_valid_nolog = prepare_train_valid_no_logs(
    snapshot_feb_sample_nolog,
    snapshot_mar_sample_nolog
)

lgbm_nolog_sample = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=63
)

lgbm_nolog_sample.fit(X_train_nolog, y_train_nolog)

y_pred_nolog = lgbm_nolog_sample.predict(X_valid_nolog)
y_prob_nolog = lgbm_nolog_sample.predict_proba(X_valid_nolog)[:, 1]

print("X_train_nolog.shape:", X_train_nolog.shape)
print("X_valid_nolog.shape:", X_valid_nolog.shape)
print("Accuracy:", accuracy_score(y_valid_nolog, y_pred_nolog))
print("F1:", f1_score(y_valid_nolog, y_pred_nolog))
print("ROC_AUC:", roc_auc_score(y_valid_nolog, y_prob_nolog))
print("LogLoss:", log_loss(y_valid_nolog, y_prob_nolog))












