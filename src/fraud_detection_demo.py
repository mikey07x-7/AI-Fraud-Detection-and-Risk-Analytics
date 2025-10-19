"""
AI Fraud Detection and Risk Analytics
Single-file runnable example that:
- Generates synthetic transaction data
- Performs cleaning & feature engineering
- Runs EDA plots
- Trains models (LightGBM + RandomForest) with SMOTE
- Evaluates performance (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Uses SHAP for interpretability
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Imbalance
from imblearn.over_sampling import SMOTE

# Explainability
import shap

# For reproducibility
RND = 42
np.random.seed(RND)
random.seed(RND)

# ---------- 1) Synthetic dataset generation ----------
def generate_synthetic_transactions(n_users=2000, n_transactions=50000, fraud_rate=0.02):
    """
    Create a synthetic transactions dataset.
    Columns:
      - transaction_id, user_id, timestamp, amount, transaction_type, device_type,
        ip_country, is_fraud (target)
    Fraud is simulated by introducing anomalies: high amount deviations, unusual device, new ip country, rapid-fire transactions.
    """
    users = [f"user_{i:05d}" for i in range(n_users)]
    # Per-user baseline behavior
    user_avg_amount = np.random.lognormal(mean=3.5, sigma=0.8, size=n_users)  # typical avg amount per user
    user_amount_std = user_avg_amount * (0.3 + np.random.rand(n_users) * 0.7)  # variability
    user_devices = np.random.choice(['mobile', 'desktop', 'tablet'], size=n_users, p=[0.6,0.3,0.1])
    user_home_country = np.random.choice(['IN','US','GB','NG','CN','BR','DE','FR'], size=n_users, p=[0.25,0.2,0.1,0.08,0.1,0.07,0.1,0.1])

    records = []
    for t_id in range(n_transactions):
        user_idx = np.random.randint(0, n_users)
        user = users[user_idx]
        base_amt = max(1.0, np.random.normal(user_avg_amount[user_idx], user_amount_std[user_idx]))
        # occasional high spikes
        if np.random.rand() < 0.02:
            amount = base_amt * np.random.uniform(5, 30)
        else:
            amount = base_amt * np.random.uniform(0.3, 3.0)

        # time: pick a timestamp across last 90 days
        ts = pd.Timestamp.now() - pd.to_timedelta(np.random.randint(0, 90*24*60), unit='m')

        # transaction type
        transaction_type = np.random.choice(['purchase', 'transfer', 'cash_withdrawal', 'payment', 'refund'], p=[0.6,0.15,0.1,0.1,0.05])

        # device and ip country — sometimes different than user's usual
        if np.random.rand() < 0.03:
            device = np.random.choice(['mobile','desktop','tablet'])
        else:
            device = user_devices[user_idx]
        if np.random.rand() < 0.05:
            ip_country = np.random.choice(['IN','US','GB','NG','CN','BR','DE','FR'])
        else:
            ip_country = user_home_country[user_idx]

        records.append({
            'transaction_id': f"tx_{t_id:07d}",
            'user_id': user,
            'timestamp': ts,
            'amount': round(float(amount),2),
            'transaction_type': transaction_type,
            'device_type': device,
            'ip_country': ip_country
        })

    df = pd.DataFrame(records)

    # Compute some derived per-user stats and add label (is_fraud)
    df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)

    # Add label with heuristics: large deviation + new device/ip + rapid sequence => higher fraud probability
    df['user_avg_amount'] = df.groupby('user_id')['amount'].transform('mean')
    df['amount_dev_ratio'] = (df['amount'] - df['user_avg_amount']).abs() / (df['user_avg_amount'] + 1e-6)

    # time since last transaction for same user
    df['ts_epoch'] = df['timestamp'].astype('int64') // 10**9
    df['ts_prev'] = df.groupby('user_id')['ts_epoch'].shift(1)
    df['time_since_last_mins'] = (df['ts_epoch'] - df['ts_prev']).fillna(999999) / 60.0

    # new device/ip flag relative to user's previous values
    df['prev_device'] = df.groupby('user_id')['device_type'].shift(1)
    df['prev_ip'] = df.groupby('user_id')['ip_country'].shift(1)
    df['device_change'] = (df['device_type'] != df['prev_device']).astype(int).fillna(0)
    df['ip_change'] = (df['ip_country'] != df['prev_ip']).astype(int).fillna(0)

    # base fraud probability (very low)
    base_prob = np.full(len(df), fraud_rate)

    # increase probabilities by heuristics
    prob = base_prob.copy()
    prob += 0.15 * (df['amount_dev_ratio'] > 2.5).astype(float)  # massive deviation
    prob += 0.12 * (df['device_change'] == 1).astype(float)
    prob += 0.12 * (df['ip_change'] == 1).astype(float)
    prob += 0.10 * (df['time_since_last_mins'] < 1).astype(float)  # rapid-fire transactions
    prob = np.clip(prob, 0, 0.9)

    df['is_fraud'] = (np.random.rand(len(df)) < prob).astype(int)

    # If class imbalance not matching target fraud_rate exactly, adjust some to reach approximate target
    actual_rate = df['is_fraud'].mean()
    # keep as generated (realistic), do not force exact rate

    # Drop helper columns we might not want raw in dataset
    df = df.drop(columns=['ts_epoch','ts_prev','prev_device','prev_ip'])

    return df

# Generate dataset
df = generate_synthetic_transactions(n_users=2500, n_transactions=60000, fraud_rate=0.01)

print("Dataset shape:", df.shape)
print("Fraud rate:", df['is_fraud'].mean())

# ---------- 2) Data cleaning & basic EDA ----------
# Quick data checks
print("\nSample rows:")
print(df.head())

# Convert timestamp to datetime (already datelike), but ensure dtype
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# EDA plots (histograms, boxplots, class imbalance)
sns.set(style="whitegrid")
plt.figure(figsize=(10,5))
sns.histplot(df['amount'], bins=100, log_scale=(False, True))
plt.title('Transaction Amount Distribution (log scale on y)')
plt.xlabel('Amount')
plt.show()

plt.figure(figsize=(10,4))
sns.boxplot(x='transaction_type', y='amount', data=df)
plt.title('Amount by Transaction Type')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='is_fraud', data=df)
plt.title('Class distribution (0 = non-fraud, 1 = fraud)')
plt.show()

# Correlation of numeric features vs label (quick)
numeric_cols = ['amount','user_avg_amount','amount_dev_ratio','time_since_last_mins','device_change','ip_change']
corr = df[numeric_cols + ['is_fraud']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation matrix (numeric features)')
plt.show()

# ---------- 3) Feature engineering ----------
def create_features(df):
    df = df.copy()
    # Transaction hour, day of week
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Transaction frequency per user: number of transactions in last 7 days (approximation using groupby)
    recent_window = pd.Timedelta(days=7)
    # For speed, compute transactions per user total and rolling per user isn't trivial in static synthetic data
    df['tx_count_user'] = df.groupby('user_id')['transaction_id'].transform('count')

    # Time since last was already available (time_since_last_mins)
    # Amount deviation (already)
    # Create features from categorical columns
    # Encode transaction_type, device_type, ip_country via label encoding (simple and OK for tree models)
    for col in ['transaction_type','device_type','ip_country']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    # Ratio of amount to user average
    df['amount_over_avg'] = df['amount'] / (df['user_avg_amount'] + 1e-6)

    # Binary flags
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['high_amount_flag'] = (df['amount_over_avg'] > 3).astype(int)

    # Drop columns not to be used raw
    drop_cols = ['transaction_id','timestamp','user_id']
    df_model = df.drop(columns=drop_cols)
    return df_model

df_feat = create_features(df)
print("\nFeature dataframe sample:")
print(df_feat.head())

# Prepare X and y
target = 'is_fraud'
X = df_feat.drop(columns=[target])
y = df_feat[target]

# Train-test split (stratify to keep distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)

print("\nTrain fraud rate:", y_train.mean(), "Test fraud rate:", y_test.mean())

# ---------- 4) Scaling / preprocessing ----------
# Tree-based models don't require scaling, but we'll standardize a few numeric features for e.g., logistic or NN if needed.
numeric_features = ['amount','user_avg_amount','amount_dev_ratio','time_since_last_mins','amount_over_avg','tx_count_user']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ---------- 5) Handle class imbalance (SMOTE for training set) ----------
print("\nBefore SMOTE, class distribution:", np.bincount(y_train))
sm = SMOTE(sampling_strategy='auto', random_state=RND)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE, class distribution:", np.bincount(y_res))

# ---------- 6) Model training: LightGBM and RandomForest ----------
# LightGBM dataset
lgb_train = lgb.Dataset(X_res, label=y_res)

# Simple LGBM parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': RND,
    'verbose': -1
}

print("\nTraining LightGBM...")
bst = lgb.train(lgb_params, lgb_train, num_boost_round=200)

# RandomForest baseline (train on resampled too)
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, n_jobs=-1)
rf.fit(X_res, y_res)

# ---------- 7) Evaluation functions ----------
def evaluate_model(model, X_t, y_t, model_name="model"):
    # If LightGBM Booster
    if isinstance(model, lgb.basic.Booster):
        y_prob = model.predict(X_t)
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = model.predict_proba(X_t)[:,1]
        y_pred = model.predict(X_t)

    precision = precision_score(y_t, y_pred, zero_division=0)
    recall = recall_score(y_t, y_pred, zero_division=0)
    f1 = f1_score(y_t, y_pred, zero_division=0)
    roc = roc_auc_score(y_t, y_prob)
    # PR AUC
    prec, rec, _ = precision_recall_curve(y_t, y_prob)
    pr_auc = auc(rec, prec)
    cm = confusion_matrix(y_t, y_pred)
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}  PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix (rows: true, cols: pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_t, y_pred, digits=4, zero_division=0))
    return {'precision':precision, 'recall':recall, 'f1':f1, 'roc_auc':roc, 'pr_auc':pr_auc, 'cm':cm, 'y_prob':y_prob, 'y_pred':y_pred}

# Evaluate on test set
print("\nEvaluating on test set...")
res_lgb = evaluate_model(bst, X_test_scaled, y_test, "LightGBM")
res_rf = evaluate_model(rf, X_test_scaled, y_test, "RandomForest")

# ---------- 8) Plot ROC and PR curves ----------
def plot_roc_pr(y_true, y_prob, model_name="Model"):
    fpr, tpr, _ = sklearn_metrics_roc(y_true, y_prob)
    prec, rec, _ = sklearn_metrics_pr(y_true, y_prob)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_true,y_prob):.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(rec, prec, label=f'{model_name} (PR-AUC={auc(rec,prec):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# small wrappers to avoid repeated imports
from sklearn.metrics import roc_curve as sklearn_metrics_roc
from sklearn.metrics import precision_recall_curve as sklearn_metrics_pr

plot_roc_pr(y_test, res_lgb['y_prob'], "LightGBM")
plot_roc_pr(y_test, res_rf['y_prob'], "RandomForest")

# ---------- 9) Business metrics: False positive / negative rates ----------
def business_metrics(cm):
    # cm is 2x2 (tn, fp; fn, tp)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {'FPR': fpr, 'FNR': fnr}

print("\nBusiness metrics (LightGBM):", business_metrics(res_lgb['cm']))
print("Business metrics (RandomForest):", business_metrics(res_rf['cm']))

# Visual: confusion matrix heatmap for the best model (choose LightGBM)
plt.figure(figsize=(5,4))
sns.heatmap(res_lgb['cm'], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (LightGBM)')
plt.show()

# ---------- 10) Interpretability: SHAP ----------
# Use TreeExplainer for tree models
print("\nComputing SHAP values (may take a moment)...")
# Prepare a smaller sample for speed in SHAP plots
X_shap_sample = X_test_scaled.sample(n=min(2000, len(X_test_scaled)), random_state=RND)

explainer = shap.TreeExplainer(bst)  # works with LightGBM
shap_values = explainer.shap_values(X_shap_sample)

# summary plot (bar)
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=True)

# detailed beeswarm (may be slow)
shap.summary_plot(shap_values, X_shap_sample, show=True)

# Example: dependence plot for amount_dev_ratio
if 'amount_dev_ratio' in X_shap_sample.columns:
    shap.dependence_plot('amount_dev_ratio', shap_values, X_shap_sample, show=True)

print("\nDone. You can save the trained models using joblib or lightgbm.save_model.")

# Save models (optional)
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/rf_model.joblib')
bst.save_model('models/lgb_model.txt')

print("Saved models to 'models/' directory.")
