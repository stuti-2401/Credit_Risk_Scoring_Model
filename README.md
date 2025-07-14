📈 Credit Risk Scoring Model
This project implements a Credit Risk Scoring pipeline using real-world Lending Club loan data.

It includes data cleaning, feature engineering, encoding categorical variables, training a Logistic Regression model, and evaluating results with ROC-AUC.

The output includes a saved ROC Curve image for visualization.

⭐️ Features
Load large Lending Club dataset

Clean and filter loan status

Handle missing values

Feature engineering (loan/income ratio, term, grade)

Encode categorical variables (emp_length, home_ownership, verification_status, purpose)

Train/test split

Train Logistic Regression model

Evaluate with Classification Report and ROC-AUC Score

Save ROC Curve plot

📂 Folder Structure
Copy
Edit
Credit_Risk_Scoring_Model/
│
├── archive/
│   └── accepted_2007_to_2018Q4.csv
├── credit_risk_model.py
├── roc_curve.png
└── README.md

⚙️ Requirements
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Recommended contents of requirements.txt:

nginx
Copy
Edit
pandas
numpy
scikit-learn
matplotlib
seaborn
🚀 How to Run
From the project root:

nginx
Copy
Edit
python credit_risk_model.py

🧭 Sample Output
ROC Curve saved as roc_curve.png

Classification Report printed in terminal

ROC-AUC Score displayed

Example Classification Report:

markdown
Copy
Edit
              precision    recall  f1-score   support

           0       0.82      0.98      0.89
           1       0.44      0.08      0.13

    accuracy                           0.80
   macro avg       0.63      0.53      0.51
weighted avg       0.74      0.80      0.74
Example ROC-AUC Score:

yaml
Copy
Edit
ROC-AUC Score: 0.69

✅ Next Steps / Ideas
Handle class imbalance with SMOTE or class_weight

Use RandomForest or XGBoost classifiers

Hyperparameter tuning with GridSearchCV

Streamlit dashboard for predictions

Cloud deployment

📜 License
This project is for educational purposes only and not for production financial decision-making.

🙏 Acknowledgments
Lending Club open data (via Kaggle)

scikit-learn, pandas, matplotlib

⭐️ Author
Stuti Sharma