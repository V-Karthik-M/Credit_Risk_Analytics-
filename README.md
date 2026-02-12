**Credit Risk Analytics – End-to-End Machine Learning + MLOps Project**

**Project Overview**
This project builds a complete end-to-end Credit Risk Analytics system that predicts whether a customer will default on credit card payments next month.
The goal was not just to train a model, but to simulate a real production ML workflow including:
Data cleaning and validation
Feature engineering
Model comparison
Experiment tracking
Deployment-style batch scoring
Drift monitoring
SQL-based operational extraction
Executive BI dashboard reporting
This mirrors how real-world ML systems are built in financial institutions.

**Dataset**
UCI Credit Card Default Dataset
30,000 customer records
23 original features
**Target variable:**
default.payment.next.month
0 = No Default
1 = Default

**Features include:**
Demographics (AGE, EDUCATION, MARRIAGE)
Credit limit (LIMIT_BAL)
Repayment history (PAY_0 to PAY_6)
Bill amounts (BILL_AMT1–6)
Payment amounts (PAY_AMT1–6)

**Data Cleaning & Validation**
Checked for missing values
Removed duplicate records
Validated numerical ranges
Created missing-value flags where applicable
Ensured clean modeling dataset
Tools: Pandas, NumPy

**Exploratory Data Analysis (EDA)**
Performed exploratory analysis to understand risk drivers:
Analyzed class imbalance (~22% defaults)
Correlation analysis with target
Identified strong predictors:
Late payments
High credit utilization
Lower repayment ratios
Visualizations built using Matplotlib.

**Feature Engineering**
Created behavioral risk indicators:
num_late_payments
avg_payment_delay
avg_bill_amount
credit_utilization
avg_payment_amount
payment_to_bill_ratio
These engineered features better capture financial stress patterns than raw columns alone.

**Model Development**
Baseline Model – Logistic Regression
Class-weight balanced for imbalanced data
Interpretable baseline model
Final Model – XGBoost
Gradient boosting algorithm
Captures nonlinear relationships
Improved ranking performance
Selected based on evaluation metrics

**Model Evaluation**
Since the dataset is imbalanced, evaluation included:
ROC-AUC
Precision
Recall
F1-score
Precision–Recall Curve
Final Model Performance:
XGBoost ROC-AUC ≈ 0.768
Stronger Precision-Recall tradeoff than Logistic Regression

**Experiment Tracking (MLflow)**
Implemented MLflow for:
Logging hyperparameters
Logging evaluation metrics
Saving model artifacts
Storing feature schema
Comparing multiple runs
This ensures reproducibility and governance.

**Deployment Simulation (Batch Scoring)**
Simulated production scoring:
Saved model using joblib
Generated customer-level risk scores
Created risk buckets:
Low
Medium
High
Stored:
model_version
scoring_date
Outputs saved in:
deploy_outputs/

**Monitoring & Drift Detection**
Implemented monitoring logic:
Prediction Drift
Compared baseline vs current risk score distribution.
Feature Drift
Monitored distribution shifts in:
num_late_payments
avg_payment_delay
credit_utilization
payment_to_bill_ratio
LIMIT_BAL
Flagged drift if distribution change exceeded threshold.
Outputs stored in:
monitoring/

**SQL-Based Operational Extraction**
Used DuckDB to simulate warehouse-style queries:
SELECT customer_id, risk_score
FROM predictions
WHERE risk_bucket = 'High'
ORDER BY risk_score DESC;
Generated high-risk customer list for operations teams.

**Executive BI Dashboard (Tableau)**
Built a Tableau dashboard including:
Customer Risk Segmentation
Risk Score Distribution
High-Risk Customer List
Model Performance Metrics
Designed for Director / VP-level reporting.

**Tech Stack**
Python
Pandas
NumPy
Scikit-learn
XGBoost
MLflow
DuckDB
Joblib
Matplotlib
Tableau
Git & GitHub
