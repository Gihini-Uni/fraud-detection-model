# Fraud Detection Model

A machine learning pipeline that flags fraudulent card, mobile, ATM and wire transactions. Built as a group project for *DA4641 Introduction to FinTech* (Case Study 01), Department of Decision Science, University of Moratuwa.

## What it does

The model reads a transaction's details and predicts whether it is fraudulent (`is_fraud = 1`) or legitimate. These details include the amount, distance from the customer's home, device used, IP risk score and account age. It outputs a fraud probability, which a bank can turn into actions such as auto-approve, step-up authentication or manual review.

## How it works

1. **Clean the data.** Remove duplicate transactions, parse mixed date formats, standardise categories and impute missing values.
2. **Engineer features.** Add signals such as transaction amount vs. the customer's 30-day average, a foreign-transaction flag, a new-account flag and a log-scaled amount.
3. **Handle class imbalance.** Only about 9.4% of transactions are fraud, so class weighting and SMOTE oversampling were compared.
4. **Train and compare four models.** Logistic Regression, Random Forest, Random Forest + SMOTE and XGBoost, all measured against a "never flag fraud" baseline.
5. **Evaluate.** Accuracy is misleading on imbalanced data, so models were judged on precision, recall, F1, ROC-AUC and PR-AUC.

## Results

**Random Forest (balanced)** was the recommended model.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | – | – |
| Logistic Regression (balanced) | 0.325 | 0.771 | 0.457 | 0.900 | 0.569 |
| **Random Forest (balanced)** | **0.509** | 0.573 | **0.539** | 0.895 | 0.531 |
| Random Forest + SMOTE | 0.472 | 0.615 | 0.534 | 0.890 | 0.508 |
| XGBoost (scale_pos_weight) | 0.496 | 0.573 | 0.531 | 0.891 | 0.538 |

Key findings:

- **Strongest fraud signal:** transaction amount (and amount relative to the customer's usual spending) dominated feature importance in both Random Forest and XGBoost.
- **Trade-off:** Logistic Regression catches the most fraud (77% recall) but raises many more false alarms (154 vs. 53 for Random Forest on the test set).
- **Threshold tuning:** moving the decision threshold from 0.7 to 0.3 nearly triples recall, so a bank can tune the model to its own cost of missed fraud vs. blocked customers.

## Repository contents

| File | Description |
|---|---|
| `FinTech_Assignment_FINAL.ipynb` | End-to-end notebook: cleaning, feature engineering, modelling, evaluation |
| `README.md` | This file |

The dataset (`dataset02.csv`) was provided by the course and is **not** included in this repository.

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (or Jupyter).
2. Add `dataset02.csv` to your Google Drive, or update the path in the data-loading cell to wherever you saved it. The notebook currently reads from `/content/drive/MyDrive/dataset02.csv`.
3. Run all cells from top to bottom.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn.

## Limitations

- Trained on a single-period synthetic dataset, so real fraud patterns would drift over time and the model would need monitoring and retraining.
- Precision stays well below 100%, so any deployment produces false positives that cost customer experience.
- Country-based features can act as proxies for national origin, so a fairness audit (for example, false-positive rates by country) is needed before real-world use.
- Random Forest's built-in importance under-weights one-hot categorical features, so the feature-importance rankings differ between models. Only the dominance of the amount features is consistent across both.

## AI-use disclosure

Claude (Anthropic) assisted with data-quality diagnosis, pipeline development and drafting, per the module's AI-use policy. All code was executed and verified by the group.
