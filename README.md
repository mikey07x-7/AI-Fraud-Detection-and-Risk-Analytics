# AI Fraud Detection and Risk Analytics

This project demonstrates how AI can automate real corporate work in detecting fraud and analyzing risk.

## 🚀 Features
- Synthetic data generation for transactions
- Feature engineering for fraud signals (amount deviation, device/IP changes, time gaps)
- Handling class imbalance using SMOTE
- Model training with LightGBM and RandomForest
- Model evaluation with Precision, Recall, F1, ROC-AUC, PR-AUC
- SHAP-based interpretability
- Professional GitHub folder structure

## 📂 Folder Structure
```
ai-fraud-detection/
├── data/
├── notebooks/
├── src/
├── models/
├── requirements.txt
└── README.md
```

## 📦 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Run the Demo
```bash
python src/fraud_detection_demo.py
```

## 📊 Next Steps
- Replace synthetic data with real anonymized data
- Tune threshold based on business cost
- Deploy using FastAPI / Flask
