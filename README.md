## 🚀 Live Demo

[Open FraudGuard Dashboard](https://fraudguard-ai-fraud-detection.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraudguard-ai-fraud-detection.streamlit.app/)
# FraudGuard — AI-Powered Financial Risk Monitoring

FraudGuard is a machine-learning-based financial risk monitoring dashboard that identifies potentially fraudulent credit card transactions and categorizes them by model risk score.

## Features

- Interactive Streamlit dashboard
- Transaction-level fraud prediction
- Low, Medium, High and Critical risk categories
- Risk analytics and visualizations
- Explainable AI using SHAP
- Downloadable CSV reports
- Batch prediction through CSV uploads

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- Pandas and NumPy
- Plotly
- SHAP
- Joblib

## Machine Learning Model

Random Forest Classifier with balanced subsample class weights.

Dataset: European Credit Card Fraud Detection dataset.

## Held-Out Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 99.96% |
| Precision | 89.25% |
| Recall | 84.69% |
| F1-score | 86.91% |
| ROC-AUC | 94.78% |
| PR-AUC | 85.97% |

Test dataset: 56,962 transactions.

These results come from a separately trained evaluation model using the same configuration.

## Installation

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

Place the required dataset and saved model files in the expected project locations before running.

## Important Notes

- Model risk scores are not calibrated fraud probabilities.
- Flagged transactions require human review.
- SHAP values explain model contributions, not causation.
- V1–V28 are anonymized PCA features.
- This project is intended for educational and demonstration purposes.

## Future Improvements

- Probability calibration
- Threshold optimization
- Real-time transaction monitoring
- Automated alerts
- Model drift monitoring

## Author

Drishty Ahlawat

B.Tech Computer Science and Engineering
