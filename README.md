Fraud Detection Model

A machine learning pipeline that flags fraudulent card, mobile, ATM and wire transactions. It was built for DA4641 Introduction to FinTech (University of Moratuwa).

What it does

The model reads a transaction’s details and predicts whether it is fraudulent (is_fraud = 1) or legitimate. These are details like the amount, how far it is from the customer’s home, the device used, the IP risk score and the account age. It outputs a fraud probability, which a bank can turn into actions such as auto-approve, step-up authentication or manual review.

How it works
Clean the data. Remove duplicate transactions, parse mixed date formats, standardise categories and impute missing values.
Engineer features. Add signals such as transaction amount vs. the customer’s 30-day average, a foreign-transaction flag, a new-account flag and a log-scaled amount.
Handle class imbalance. Only about 9.4% of transactions are fraud, so class weighting and SMOTE oversampling were compared.
Train and compare four models. These were Logistic Regression, Random Forest, Random Forest + SMOTE and XGBoost, all measured against a “never flag fraud” baseline.
Evaluate. Accuracy is misleading on imbalanced data, so the models were judged on precision, recall, F1, ROC-AUC and PR-AUC.
Results

Random Forest (balanced) was the recommended model:

Precision	Recall	F1	ROC-AUC	PR-AUC
0.509	0.573	0.539	0.895	0.531
Strongest fraud signal: transaction amount, which dominated feature importance in both Random Forest and XGBoost.
Trade-off: Logistic Regression catches the most fraud (77% recall) but raises many more false alarms.
Threshold tuning: the decision threshold can be moved to trade catching more fraud against blocking more legitimate customers.
Limitations

It was trained on a single-period synthetic dataset, so real fraud patterns would drift over time. Country-based features could act as proxies for national origin, which calls for a fairness audit before any real deployment.

Tech stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib and seaborn.
