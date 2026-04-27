################################################
# KKBox Churn Prediction - Data Loading & Inspection
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
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return na_columns
def quick_look(dataframe, head=5):
    print(dataframe.head(head))
    print(f"\nShape: {dataframe.shape}")
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def kkbox_data_prep(dataframe):
    df = dataframe.copy()

    y = df["is_churn"]

    drop_cols = [
        "is_churn",
        "msno",
        "TRANSACTION_DATE_MAX",
        "TRANSACTION_DATE_MIN",
        "MEMBERSHIP_EXPIRE_DATE_MAX",
        "registration_init_time",
        "bd",
        "NEW_AGE_CAT"
    ]

    cat_cols = [
        "PAYMENT_METHOD_ID_LAST",
        "gender",
        "registered_via",
        "city"
    ]

    df = df.drop(drop_cols, axis=1)

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    X = df

    return X, y

################################################
# Data Loading
################################################

train = pd.read_csv("data/raw/train_v2.csv")
transactions = pd.read_csv("data/raw/transactions_v2.csv")
members = pd.read_csv("data/raw/members_v3.csv")
sample_submission = pd.read_csv("data/raw/sample_submission_v2.csv")


quick_look(train)
quick_look(transactions)
quick_look(members)
quick_look(sample_submission)


train["msno"].is_unique
train["msno"].nunique(), train.shape[0]

members["msno"].is_unique
members["msno"].nunique(), members.shape[0]

transactions["msno"].is_unique
transactions["msno"].nunique(), transactions.shape[0]

train["msno"].duplicated().sum()
members["msno"].duplicated().sum()
transactions["msno"].duplicated().sum()

transactions["msno"].value_counts().head(10)


quick_look(transactions)


################################################
# Transactions Feature Table
################################################

transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], format="%Y%m%d")
transactions["membership_expire_date"] = pd.to_datetime(transactions["membership_expire_date"], format="%Y%m%d")

transactions = transactions.sort_values(["msno", "transaction_date"])

transactions_agg = transactions.groupby("msno").agg({
    "payment_method_id": ["nunique", "last"],
    "payment_plan_days": ["mean", "last"],
    "plan_list_price": ["mean", "last"],
    "actual_amount_paid": ["mean", "last"],
    "is_auto_renew": ["mean", "last"],
    "is_cancel": ["sum", "mean", "last"],
    "transaction_date": ["max", "min"],
    "membership_expire_date": ["max"]
})

transactions_agg.columns = ["_".join(col).upper() for col in transactions_agg.columns]
transactions_agg = transactions_agg.reset_index()

transactions_agg.head()
transactions_agg.shape


################################################
# Members Preparation
################################################

members["registration_init_time"] = pd.to_datetime(members["registration_init_time"], format= "%Y%m%d")

members.head()
quick_look(members)
members['msno'].is_unique

members.shape
transactions_agg.shape

################################################
# Merge: Transactions + Members
################################################

df = transactions_agg.merge(members, how='left', on='msno')

df.head()
df.shape
df['msno'].is_unique

################################################
# Final Merge: Train + Feature Table
################################################

df = train.merge(df, how='left', on='msno')

df.head()
df.shape
df['msno'].is_unique

################################################
# Final Data Checks
################################################

check_df(df)

df['is_churn'].value_counts()
df['is_churn'].value_counts(normalize=True)

missing_values_table(df)

quick_look(df)

################################################
# Feature Engineering
################################################

df["bd"].describe()
df["bd"].value_counts().sort_index().head(30)
df["bd"].value_counts().sort_index().tail(30)

df.loc[df["bd"] <= 15, "bd"].value_counts().sort_index()
df.loc[df["bd"] >= 70, "bd"].value_counts().sort_index()

# Age cleaning
df["bd"] = df["bd"].astype(float)
df.loc[(df["bd"] >= 1900) & (df["bd"] <= 2017), "bd"] = 2017 - df["bd"]
df.loc[(df["bd"] < 10) | (df["bd"] > 100), "bd"] = pd.NA

df["bd"].describe()
df["bd"].isnull().sum()

# Missing flags
df["NEW_NO_TRANSACTION"] = df["TRANSACTION_DATE_MAX"].isnull().astype(int)
df["NEW_NO_MEMBER_INFO"] = df["registration_init_time"].isnull().astype(int)
df["NEW_GENDER_MISSING"] = df["gender"].isnull().astype(int)

# Date-based features
df["NEW_MEMBERSHIP_DURATION_DAYS"] = (df["MEMBERSHIP_EXPIRE_DATE_MAX"] - df["TRANSACTION_DATE_MIN"]).dt.days
df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = (df["MEMBERSHIP_EXPIRE_DATE_MAX"] - df["TRANSACTION_DATE_MAX"]).dt.days
df["NEW_REG_TO_LAST_TRANS_DAYS"] = (df["TRANSACTION_DATE_MAX"] - df["registration_init_time"]).dt.days

# Payment-related features
df["NEW_PRICE_DIFF_LAST"] = df["PLAN_LIST_PRICE_LAST"] - df["ACTUAL_AMOUNT_PAID_LAST"]
df["NEW_PRICE_DIFF_MEAN"] = df["PLAN_LIST_PRICE_MEAN"] - df["ACTUAL_AMOUNT_PAID_MEAN"]

# Behaviour features
df["NEW_IS_DISCOUNT_USER"] = (df["ACTUAL_AMOUNT_PAID_MEAN"] < df["PLAN_LIST_PRICE_MEAN"]).astype(float)
df["NEW_CANCEL_RATE"] = df["IS_CANCEL_MEAN"]
df["NEW_AUTO_RENEW_RATE"] = df["IS_AUTO_RENEW_MEAN"]

# Simple age categories
df.loc[df["bd"] < 25, "NEW_AGE_CAT"] = "young"
df.loc[(df["bd"] >= 25) & (df["bd"] < 40), "NEW_AGE_CAT"] = "adult"
df.loc[(df["bd"] >= 40) & (df["bd"] < 60), "NEW_AGE_CAT"] = "middle_age"
df.loc[df["bd"] >= 60, "NEW_AGE_CAT"] = "senior"


df[[
    "bd",
    "NEW_NO_TRANSACTION",
    "NEW_NO_MEMBER_INFO",
    "NEW_GENDER_MISSING",
    "NEW_MEMBERSHIP_DURATION_DAYS",
    "NEW_LAST_TRANS_TO_EXPIRE_DAYS",
    "NEW_REG_TO_LAST_TRANS_DAYS",
    "NEW_PRICE_DIFF_LAST",
    "NEW_PRICE_DIFF_MEAN",
    "NEW_IS_DISCOUNT_USER",
    "NEW_CANCEL_RATE",
    "NEW_AUTO_RENEW_RATE",
    "NEW_AGE_CAT"
]].head(10)

missing_values_table(df)


################################################
# Feature Fixes
################################################

df.loc[df["NEW_NO_TRANSACTION"] == 1, "NEW_IS_DISCOUNT_USER"] = pd.NA

df.loc[df["NEW_MEMBERSHIP_DURATION_DAYS"] < 0, "NEW_MEMBERSHIP_DURATION_DAYS"] = pd.NA
df.loc[df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] < 0, "NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = pd.NA
df.loc[df["NEW_REG_TO_LAST_TRANS_DAYS"] < 0, "NEW_REG_TO_LAST_TRANS_DAYS"] = pd.NA


df[[
    "NEW_NO_TRANSACTION",
    "NEW_MEMBERSHIP_DURATION_DAYS",
    "NEW_LAST_TRANS_TO_EXPIRE_DAYS",
    "NEW_REG_TO_LAST_TRANS_DAYS",
    "NEW_IS_DISCOUNT_USER"
]].describe()

(df["NEW_MEMBERSHIP_DURATION_DAYS"] < 0).sum()
(df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] < 0).sum()
(df["NEW_REG_TO_LAST_TRANS_DAYS"] < 0).sum()

missing_values_table(df)

df.head()

######################################################

######################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

cat_cols
num_cols
cat_but_car
len(cat_cols), len(num_cols), len(cat_but_car)


drop_cols = [
    "msno",
    "TRANSACTION_DATE_MAX",
    "TRANSACTION_DATE_MIN",
    "MEMBERSHIP_EXPIRE_DATE_MAX",
    "registration_init_time"
]

manual_cat_cols = [
    "PAYMENT_METHOD_ID_NUNIQUE",
    "PAYMENT_METHOD_ID_LAST",
    "IS_AUTO_RENEW_LAST",
    "IS_CANCEL_SUM",
    "IS_CANCEL_LAST",
    "gender",
    "registered_via",
    "city",
    "NEW_NO_TRANSACTION",
    "NEW_NO_MEMBER_INFO",
    "NEW_GENDER_MISSING",
    "NEW_IS_DISCOUNT_USER",
    "NEW_AGE_CAT"
]

manual_num_cols = [
    "PAYMENT_PLAN_DAYS_MEAN",
    "PAYMENT_PLAN_DAYS_LAST",
    "PLAN_LIST_PRICE_MEAN",
    "PLAN_LIST_PRICE_LAST",
    "ACTUAL_AMOUNT_PAID_MEAN",
    "ACTUAL_AMOUNT_PAID_LAST",
    "IS_AUTO_RENEW_MEAN",
    "IS_CANCEL_MEAN",
    "bd",
    "NEW_MEMBERSHIP_DURATION_DAYS",
    "NEW_LAST_TRANS_TO_EXPIRE_DAYS",
    "NEW_REG_TO_LAST_TRANS_DAYS",
    "NEW_PRICE_DIFF_LAST",
    "NEW_PRICE_DIFF_MEAN",
    "NEW_CANCEL_RATE",
    "NEW_AUTO_RENEW_RATE"
]

set(manual_cat_cols) & set(manual_num_cols)
[col for col in drop_cols if col in manual_cat_cols or col in manual_num_cols]


X, y = kkbox_data_prep(df)
X.shape, y.shape
X.head()
X.isnull().sum().sort_values(ascending=False).head(20)
y.value_counts()

############################################
# DATA CHECK
############################################

X['bd'].isnull().sum()
X['bd'].shape

X.isnull().sum()

na_cols = X.columns[X.isnull().sum() > 0].tolist()
na_cols
X[na_cols].isnull().sum().sort_values(ascending=False)

transaction_na_check = [
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
    "NEW_MEMBERSHIP_DURATION_DAYS",
    "NEW_LAST_TRANS_TO_EXPIRE_DAYS",
    "NEW_PRICE_DIFF_LAST",
    "NEW_PRICE_DIFF_MEAN",
    "NEW_IS_DISCOUNT_USER",
    "NEW_CANCEL_RATE",
    "NEW_AUTO_RENEW_RATE"
]
for col in transaction_na_check:
    print(col, ((X[col].isnull()) == (X["NEW_NO_TRANSACTION"] == 1)).all())

X["bd"].isnull().mean()
y.groupby(X["bd"].isnull()).mean()

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
df[transaction_fill_zero_cols].isnull().sum()


X, y = kkbox_data_prep(df)
X.shape, y.shape
X.isnull().sum().sort_values(ascending=False).head(20)


print(((X["NEW_REG_TO_LAST_TRANS_DAYS"].isnull()) == (X["NEW_NO_MEMBER_INFO"] == 1)).all())
print(((X["NEW_MEMBERSHIP_DURATION_DAYS"].isnull()) == (X["NEW_NO_TRANSACTION"] == 1)).all())
print(((X["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].isnull()) == (X["NEW_NO_TRANSACTION"] == 1)).all())

X.loc[X["NEW_REG_TO_LAST_TRANS_DAYS"].isnull(), "NEW_NO_MEMBER_INFO"].value_counts(dropna=False)
X.loc[X["NEW_REG_TO_LAST_TRANS_DAYS"].isnull(), "NEW_NO_TRANSACTION"].value_counts(dropna=False)
X.loc[X["NEW_MEMBERSHIP_DURATION_DAYS"].isnull(), "NEW_NO_TRANSACTION"].value_counts(dropna=False)
X.loc[X["NEW_MEMBERSHIP_DURATION_DAYS"].isnull(), "NEW_NO_MEMBER_INFO"].value_counts(dropna=False)
X.loc[X["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].isnull(), "NEW_NO_TRANSACTION"].value_counts(dropna=False)
X.loc[X["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].isnull(), "NEW_NO_MEMBER_INFO"].value_counts(dropna=False)


df["NEW_MEMBERSHIP_DURATION_DAYS"] = df["NEW_MEMBERSHIP_DURATION_DAYS"].fillna(0)
df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"] = df["NEW_LAST_TRANS_TO_EXPIRE_DAYS"].fillna(0)
df["NEW_REG_TO_LAST_TRANS_DAYS"] = df["NEW_REG_TO_LAST_TRANS_DAYS"].fillna(df["NEW_REG_TO_LAST_TRANS_DAYS"].median())

df[[
    "NEW_MEMBERSHIP_DURATION_DAYS",
    "NEW_LAST_TRANS_TO_EXPIRE_DAYS",
    "NEW_REG_TO_LAST_TRANS_DAYS"
]].isnull().sum()

X, y = kkbox_data_prep(df)
X.isnull().sum().sort_values(ascending=False).head(20)
X.shape, y.shape
X.head()


#####################################################################
##BASE MODEL
#####################################################################

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


def base_models(X, y):
    print("Base Models....")

    classifiers = [
        ('LR', LogisticRegression(max_iter=1000)),
        ('KNN', KNeighborsClassifier()),
        ("SVC", SVC()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ('Adaboost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier())
    ]

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring, n_jobs=-1)

        print(f"########## {name} ##########")
        print(f"Accuracy:  {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"Recall:    {round(cv_results['test_recall'].mean(), 4)}")
        print(f"F1:        {round(cv_results['test_f1'].mean(), 4)}")
        print(f"ROC_AUC:   {round(cv_results['test_roc_auc'].mean(), 4)}")
        print()
base_models(X, y)

#################################################
##SAMPLE DATASET
#################################################

X_small = X.sample(n=100000, random_state=42)
y_small = y.loc[X_small.index]

def base_models_fast(X, y):
    print("Base Models....")

    classifiers = [
        ('LR', LogisticRegression(max_iter=1000)),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('LightGBM', LGBMClassifier(random_state=42, n_jobs=-1))
    ]

    scoring = ["accuracy", "f1", "roc_auc", "neg_log_loss"]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring, n_jobs=-1)

        print(f"########## {name} ##########")
        print(f"Accuracy:      {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"F1:            {round(cv_results['test_f1'].mean(), 4)}")
        print(f"ROC_AUC:       {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"LogLoss:       {round(-cv_results['test_neg_log_loss'].mean(), 5)}")
        print()
base_models_fast(X_small, y_small)

####################################################
##HOLDOUT SPLIT AND MODELING
####################################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1)
lgbm_model.fit(X_train, y_train)

y_pred = lgbm_model.predict(X_valid)
y_prob = lgbm_model.predict_proba(X_valid)[:, 1]

print("Accuracy:", accuracy_score(y_valid, y_pred))
print("F1:", f1_score(y_valid, y_pred))
print("ROC_AUC:", roc_auc_score(y_valid, y_prob))
print("LogLoss:", log_loss(y_valid, y_prob))


lr_model = LogisticRegression(max_iter=3000, solver="liblinear")

cv_results = cross_validate(
    lr_model,
    X,
    y,
    cv=3,
    scoring=["accuracy", "f1", "roc_auc"],
    n_jobs=-1
)

print("Accuracy:", cv_results["test_accuracy"].mean())
print("F1:", cv_results["test_f1"].mean())
print("ROC_AUC:", cv_results["test_roc_auc"].mean())



xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_valid)
y_prob = xgb_model.predict_proba(X_valid)[:, 1]

print("Accuracy:", accuracy_score(y_valid, y_pred))
print("F1:", f1_score(y_valid, y_pred))
print("ROC_AUC:", roc_auc_score(y_valid, y_prob))
print("LogLoss:", log_loss(y_valid, y_prob))



gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)

y_pred = gbm_model.predict(X_valid)
y_prob = gbm_model.predict_proba(X_valid)[:, 1]

print("Accuracy:", accuracy_score(y_valid, y_pred))
print("F1:", f1_score(y_valid, y_pred))
print("ROC_AUC:", roc_auc_score(y_valid, y_prob))
print("LogLoss:", log_loss(y_valid, y_prob))




lgbm_params = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "max_depth": [-1, 10]
}

lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1)

lgbm_gs = GridSearchCV(
    estimator=lgbm_model,
    param_grid=lgbm_params,
    scoring="neg_log_loss",
    cv=3,
    n_jobs=-1,
    verbose=1
)

lgbm_gs.fit(X_train, y_train)


lgbm_best = lgbm_gs.best_estimator_

y_pred = lgbm_best.predict(X_valid)
y_prob = lgbm_best.predict_proba(X_valid)[:, 1]

print("Best Params:", lgbm_gs.best_params_)
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("F1:", f1_score(y_valid, y_pred))
print("ROC_AUC:", roc_auc_score(y_valid, y_prob))
print("LogLoss:", log_loss(y_valid, y_prob))


###########################################################
#1) confusion matrix
#2) classification report
#3) feature importance
###########################################################

from sklearn.metrics import confusion_matrix, classification_report

y_pred = lgbm_best.predict(X_valid)
y_prob = lgbm_best.predict_proba(X_valid)[:, 1]

print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))


feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": lgbm_best.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.head(20)


##########################################################
## FIGURES
##########Confusion Matrix
##########Top Feature Importances
##########ROC Curve
##########################################################

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_valid, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, values_format='d')
plt.title("LightGBM Confusion Matrix")
plt.show()




feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": lgbm_best.feature_importances_
}).sort_values("Importance", ascending=False)

top_n = 20
fi_top = feature_importance.head(top_n).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(fi_top["Feature"], fi_top["Importance"])
plt.title(f"Top {top_n} Feature Importances - LightGBM")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()




from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
roc_auc = roc_auc_score(y_valid, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


###########################################
# SAVING
###########################################

import joblib

joblib.dump(lgbm_best, "lgbm_kkbox_churn.pkl")
feature_importance.to_csv("feature_importance_lgbm.csv", index=False)

results = {
    "model": "LightGBM",
    "accuracy": accuracy_score(y_valid, y_pred),
    "f1": f1_score(y_valid, y_pred),
    "roc_auc": roc_auc_score(y_valid, y_prob),
    "log_loss": log_loss(y_valid, y_prob)
}

print(results)


##############################################################
## LEAKAGE CHEKING
##############################################################


transactions["transaction_date"].min(), transactions["transaction_date"].max()
transactions["membership_expire_date"].min(), transactions["membership_expire_date"].max()
train.shape, transactions.shape


################################################################
## transactions_cutoff
################################################################

transactions_cutoff = transactions[transactions["transaction_date"] <= "2017-02-28"].copy()

transactions_cutoff.shape
transactions_cutoff["transaction_date"].min(), transactions_cutoff["transaction_date"].max()

transactions_cutoff = transactions_cutoff.sort_values(["msno", "transaction_date"])


transactions_agg_cutoff = transactions_cutoff.groupby("msno").agg({
    "payment_method_id": ["nunique", "last"],
    "payment_plan_days": ["mean", "last"],
    "plan_list_price": ["mean", "last"],
    "actual_amount_paid": ["mean", "last"],
    "is_auto_renew": ["mean", "last"],
    "is_cancel": ["sum", "mean", "last"],
    "transaction_date": ["max", "min"],
    "membership_expire_date": ["max"]
})

transactions_agg_cutoff.columns = ["_".join(col).upper() for col in transactions_agg_cutoff.columns]
transactions_agg_cutoff = transactions_agg_cutoff.reset_index()

transactions_agg_cutoff.shape





















