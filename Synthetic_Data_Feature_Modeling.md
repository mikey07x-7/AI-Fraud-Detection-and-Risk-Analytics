# AI Fraud Detection - Data Generation, Feature Engineering & Modeling

## Synthetic Data Generation

We created many users and simulated transactions for each user. For each transaction, we generated:

* Amount
* Timestamp
* Device
* IP country
* Transaction type

Realistic anomalies were introduced, such as:

* Very large transaction amounts relative to a user's average
* Device or IP changes
* Many transactions in a short period

These anomalies increase the probability that a transaction is fraudulent.

**Why:** Real fraud detection projects often start with cleaned, anonymized, or synthetic data for prototyping.

## Data Cleaning & Quick Checks

* Checked data types, missing values, and basic distributions.
* Plotted histograms and boxplots to inspect outliers and spread of transaction amounts and types.

**Why:** Exploring the data helps locate issues, understand imbalance, and choose sensible features.

## Feature Engineering

Created features to help detect anomalies:

* `amount_dev_ratio`: deviation from the user's normal transaction amount
* `time_since_last_mins`: short times between transactions may indicate bots
* `device_change` & `ip_change`: whether device/IP is new for the user
* `hour` & `dayofweek`: capture time-based patterns
* `amount_over_avg` & `high_amount_flag`

Categorical variables (`device`, `ip_country`, `transaction_type`) were label encoded for tree-based models.

## Train-Test Split

* Split data into training and test sets while preserving the fraud proportion (stratified).
* Ensures realistic model evaluation.

## Handle Class Imbalance (SMOTE)

* Fraud is rare; classifiers may ignore the minority class.
* Applied SMOTE to create synthetic examples of fraud in the training set.
* **Important:** Apply SMOTE only on training data, not on test data.

## Model Training

Trained two models on the resampled data:

* **LightGBM**: gradient boosting, strong for tabular data
* **RandomForest**: solid baseline

## Evaluation

Metrics used:

* **Precision:** fraction of predicted frauds that are truly fraud (reduces false positives)
* **Recall:** fraction of actual frauds detected (reduces losses)
* **F1:** balances precision and recall
* **PR-AUC:** critical for imbalanced datasets
* **Confusion Matrix:** shows false positives vs false negatives

## Business Metrics

* Derived **False Positive Rate** and **False Negative Rate** from confusion matrix
* Quantifies operational impact (legitimate transactions blocked vs frauds missed)

## Explainability

* Used **SHAP** to understand which features drive predictions
* SHAP enables investigators to trust model output, ensures compliance, and aids debugging

## Save Models

* Trained models saved to `models/` directory for deployment.

## Tips & Practical Considerations (Production)

* **Data privacy:** Ensure PII protection and regulatory compliance (GDPR, PCI-DSS)
* **Feature freshness:** Time-dependent features need fast feature stores or streaming aggregation
* **Latency:** For real-time scoring, prefer fast models or precomputed features
* **Threshold tuning:** Adjust classification threshold to meet business constraints
* **Monitoring:** Track model drift, feature shifts, and maintain human-in-the-loop feedback
* **Ensemble & stacking:** Combine models for better performance and stability
