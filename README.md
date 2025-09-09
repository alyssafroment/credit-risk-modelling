# Credit Risk Modelling Analysis
This project explores the process of predicting credit card client defaults using historical repayment and account behaviour data through machine learning models.

## Objective
Predict the likelihood of a client defaulting in the next month based on their client profile. Key goals include:
- Identifying clients at risk of default
- Testing and comparing three seperate machine learning models
- Evaluating model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC
- Demonstrating hyperparameter tuning to improve prediction results
- Applying the model to new client data to illustrate its performance and potential use

## Tools Used
- Python (pandas, scikit-learn, xgboost, matplotlib, seaborn)
- Jupyter Notebook

## Features
The dataset includes features such as:
Credit_Limit – Credit limit of the client
AGE – Client age
Repayment Status (RS_Sept) –  repayment history in most recent momth (September)
Bill Amount (BA_Sept) – Amount of bill statement in most recent month (September)
Payment Amounts (PA_Sept … PA_April) – Past 6 month payments
RS_avg, BA_avg, RS_late_count – Aggregated features for client history
default_in_oct - target feature, binary client default in October

## Models
Logistic Regression – Basic linear model for classification
Random Forest – Ensemble of decision trees, tuned to improve recall
XGBoost – Gradient boosting model, tuned for best performance in capturing defaulters

## Key Findings
- Hyperparameter tuning improved both Random Forest and XGBoost model recall, which is critical for identifying clients at risk of default.
- Evaluation metrics were evaluated on all trained models: accuracy, precision, recall, F1-score, ROC_AUC
- The tuned XGBoost model performed best for recall, effectively capturing high-risk clients while maintaining reasonable precision.
- Example generated predictions for new clients show the model can estimate default probability and assign predicted default classes.

## Dataset
Data used is from Kaggle:  
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

## Notebook
All code and visualizations are included in the Jupyter Notebook in this repository.
