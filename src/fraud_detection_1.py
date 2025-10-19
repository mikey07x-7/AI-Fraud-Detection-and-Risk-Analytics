"""
AI Fraud Detection and Risk Analytics (CSV-ready)
- Loads your CSV with columns: user_id, amount, time_delta, device_type, ip_region, is_fraud
- Performs cleaning & feature engineering
- Handles class imbalance with SMOTE
- Trains LightGBM + RandomForest
- Evaluates performance (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Uses SHAP for interpretability
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix, classification_report)
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import joblib

# For reproducibility
RND = 42
np.random.seed(RND)

# ---------- 1) Load CSV ----------
df = pd.read_csv('data/synthetic_transactions.csv')  # Replace with your file path

# Quick check
print("Dataset shape:", df.shape)
print("Fraud rate:", df['is_fraud'].mean())
print(df.head())

# ---------- 2) Feature Engineering ----------
def create_features_csv(df):
    df = df.copy()

    # Label encode categorical columns
    categorical_cols = ['user_id', 'device_type', 'ip_region']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    # Derived features
    df['amount_over_avg'] = df['amount'] / (df['amount'].mean() + 1e-6)
    df['high_amount_flag'] = (df['amount_over_avg'] > 3).astype(int)

    # Drop raw categorical columns
    drop_cols = ['user_id','device_type','ip_region']
    df_model = df.drop(columns=drop_cols)
    return df_model

df_feat = create_features_csv(df)
print("\nFeature dataframe sample:")
print(df_feat.head())

# Prepare X and y
target = 'is_fraud'
X = df_feat.drop(columns=[target])
y = df_feat[target]

# ---------- 3) Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RND
)
print("\nTrain fraud rate:", y_train.mean(), "Test fraud rate:", y_test.mean())

# ---------- 4) Scaling ----------
numeric_features = ['amount','time_delta','amount_over_avg']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ---------- 5) Handle class imbalance ----------
print("\nBefore SMOTE, class distribution:", np.bincount(y_train))
sm = SMOTE(random_state=RND)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE, class distribution:", np.bincount(y_res))

# ---------- 6) Model Training ----------
# LightGBM
lgb_train = lgb.Dataset(X_res, label=y_res)
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

# RandomForest
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, n_jobs=-1)
rf.fit(X_res, y_res)

# ---------- 7) Evaluation ----------
def evaluate_model(model, X_t, y_t, model_name="model"):
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

# Evaluate
res_lgb = evaluate_model(bst, X_test_scaled, y_test, "LightGBM")
res_rf = evaluate_model(rf, X_test_scaled, y_test, "RandomForest")

# ---------- 8) ROC & PR plots ----------
from sklearn.metrics import roc_curve, precision_recall_curve
def plot_roc_pr(y_true, y_prob, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
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

plot_roc_pr(y_test, res_lgb['y_prob'], "LightGBM")
plot_roc_pr(y_test, res_rf['y_prob'], "RandomForest")

# ---------- 9) Business metrics ----------
def business_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {'FPR': fpr, 'FNR': fnr}

print("\nBusiness metrics (LightGBM):", business_metrics(res_lgb['cm']))
print("Business metrics (RandomForest):", business_metrics(res_rf['cm']))

# Confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(res_lgb['cm'], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (LightGBM)')
plt.show()

# ---------- 10) SHAP Interpretability ----------
print("\nComputing SHAP values...")
X_shap_sample = X_test_scaled.sample(n=min(2000, len(X_test_scaled)), random_state=RND)
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_shap_sample)
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=True)
shap.summary_plot(shap_values, X_shap_sample, show=True)

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/rf_model.joblib')
bst.save_model('models/lgb_model.txt')
print("Saved models to 'models/' directory.")
