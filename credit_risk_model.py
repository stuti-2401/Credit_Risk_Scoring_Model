#!/usr/bin/env python
# ---------------------------------------
# CREDIT RISK SCORING MODEL
# ---------------------------------------
# Author: Your Name
# Description:
#   Loads Lending Club loan data,
#   cleans and engineers features,
#   encodes categorical variables,
#   trains a logistic regression model,
#   evaluates with ROC-AUC,
#   and saves an ROC curve plot.
# ---------------------------------------

# =======================================
# 1. Import Libraries
# =======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# =======================================
# 2. Load Data
# =======================================
print("\n🟣 [INFO] Loading CSV...")
try:
    df = pd.read_csv(
        r'C:\Users\91981\Desktop\Credit_Risk_Scoring_Model\archive\accepted_2007_to_2018Q4.csv',
        nrows=100000
    )
except FileNotFoundError:
    print("🔴 [ERROR] CSV file not found. Check your path!")
    sys.exit(1)

print(f"✅ Loaded data shape: {df.shape}")

# =======================================
# 3. Filter Loan Status
# =======================================
print("\n🟣 [INFO] Filtering loan_status...")
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['target'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
print(f"✅ After filtering: {df.shape}")

# =======================================
# 4. Select Relevant Columns
# =======================================
print("\n🟣 [INFO] Selecting relevant columns...")
columns_of_interest = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'dti', 'delinq_2yrs',
    'open_acc', 'revol_util', 'total_acc', 'target'
]
df = df[columns_of_interest]

# =======================================
# 5. Drop Missing Values
# =======================================
print("\n🟣 [INFO] Dropping rows with missing values...")
df = df.dropna()
print(f"✅ After dropna: {df.shape}")

# =======================================
# 6. Feature Engineering
# =======================================
print("\n🟣 [INFO] Engineering new features...")

# Loan to Income Ratio
df['loan_income_ratio'] = df['loan_amnt'] / df['annual_inc']

# Term as numeric
df['term_num'] = df['term'].apply(lambda x: int(x.strip().split()[0]))

# Grade as numeric
grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
df['grade_num'] = df['grade'].map(grade_map)

# Drop original text columns no longer needed
df = df.drop(['term', 'grade'], axis=1)

print("✅ Feature Engineering Done.")

# =======================================
# 7. Clean Categorical Columns
# =======================================
print("\n🟣 [INFO] Converting categorical columns...")

def clean_emp_length(val):
    if isinstance(val, str):
        if '10+' in val:
            return 10
        elif '< 1' in val:
            return 0
        elif 'n/a' in val.lower():
            return 0
        else:
            try:
                return int(val.strip().split()[0])
            except:
                return 0
    return 0

df['emp_length'] = df['emp_length'].apply(clean_emp_length)

# One-hot encode other categorical columns
print("🟣 [INFO] One-hot encoding categorical columns...")
df = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)

print("✅ Final columns after encoding:")
print(df.columns.tolist())

# =======================================
# 8. Define Features and Target
# =======================================
X = df.drop('target', axis=1)
y = df['target']

# =======================================
# 9. Split into Train and Test Sets
# =======================================
print("\n🟣 [INFO] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train shape: {X_train.shape}")
print(f"✅ Test shape: {X_test.shape}")

# =======================================
# 10. Train Logistic Regression Model
# =======================================
print("\n🟣 [INFO] Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =======================================
# 11. Make Predictions
# =======================================
print("\n🟣 [INFO] Making predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =======================================
# 12. Evaluate the Model
# =======================================
print("\n✅ ===== Classification Report =====")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\n✅ ROC-AUC Score: {roc_auc:.4f}")

# =======================================
# 13. Plot and Save ROC Curve
# =======================================
print("\n🟣 [INFO] Plotting and saving ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Risk Scoring Model')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

print("✅ ROC Curve saved as 'roc_curve.png'")
print("\n🎯 [INFO] Script completed successfully.")
