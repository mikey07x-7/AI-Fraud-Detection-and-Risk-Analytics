# AI Fraud Detection and Risk Analytics

This project demonstrates how AI can automate real corporate work in detecting fraud and analyzing risk.

---

## 🚀 Features

- Synthetic data generation for transactions  
- Feature engineering for fraud signals (amount deviation, device/IP changes, time gaps)  
- Handling class imbalance using SMOTE  
- Model training with LightGBM and RandomForest  
- Model evaluation with Precision, Recall, F1, ROC-AUC, PR-AUC  
- SHAP-based interpretability  
- Professional GitHub folder structure  

---

## 📂 Folder Structure

ai-fraud-detection/
├── data/                      # Your CSV dataset
│   └── transactions.csv
├── models/                    # Trained models saved here
├── src/                       # Source code scripts
│   └── fraud_detection_demo.py
├── notebooks/                 # Optional Jupyter notebooks
├── requirements.txt           # Python dependencies
├── README.md                  # This file



yaml
Copy code

---

## 📦 Installation

```bash
pip install -r requirements.txt
▶️ Run the Demo
bash
Copy code
python src/fraud_detection_demo.py
This will:

Generate synthetic transaction data

Perform feature engineering

Train LightGBM and RandomForest models

Evaluate performance metrics

Produce SHAP interpretability plots

Save models to the models/ folder

📊 Next Steps
Replace synthetic data with real anonymized transaction data

Run src/fraud_detection_1.py with the new data.csv ( change the path in code )

Tune classification thresholds based on business cost

Deploy using FastAPI or Flask for real-time prediction
