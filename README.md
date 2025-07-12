# Credit Risk Analysis

## Project Overview

The **Credit Risk Analysis** project aims to develop and evaluate machine learning models that predict the likelihood of loan approval or default based on borrower financial data. This model helps financial institutions streamline loan approval processes, reduce subjectivity, and minimize risk by providing data-driven predictions.

The dataset includes various borrower attributes, such as income percentage, credit length, loan intent, home ownership status, and interest rate, which are used to build and compare multiple classification models.

---

## Features & Goals

- Automate loan approval risk assessment
- Handle imbalanced datasets using SMOTE
- Fill missing values using KNN imputer
- Feature selection to improve model performance
- Train and compare classifiers like Logistic Regression, Decision Trees, Random Forest, XGBoost, SVM, Naive Bayes, and ensemble methods like stacking and bagging
- Evaluate models using precision, recall, F1-score, ROC curves, and cross-validation

---

## Technologies & Libraries

- Python 3.x
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn, Plotly (visualization)
- Scikit-learn (machine learning models and utilities)
- imbalanced-learn (SMOTE for balancing data)
- XGBoost (gradient boosting)
- Flask (for deployment - if applicable)

---

## Project Structure
```
Credit_risk_analysis/
│
├── data/
│ └── credit_risk.csv # Raw dataset file
│
├── notebooks/
│ └── exploratory_data_analysis.ipynb # Initial EDA and visualization
│
├── src/
│ ├── data_preprocessing.py # Scripts for data cleaning, imputation, encoding
│ ├── feature_selection.py # Feature selection code using SequentialFeatureSelector
│ ├── model_training.py # Training and tuning ML models
│ ├── evaluation.py # Model evaluation metrics and visualization
│ └── utils.py # Utility functions
│
├── models/
│ └── final_rf_model.pkl # Example saved model
│
├── app.py # Flask app for deployment (optional)
├── requirements.txt # Required Python packages
└── README.md # This documentation file
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Lawrence302/credit_risk_analysis.git
   cd credit_risk_analysis
  ```
2. Install required packages
```bash
pip install -r requirements.txt
```
3. Run data preprocessing and exploratory data analysis from the notebooks.

4. Train models and evaluate their performance using the scripts in src/.

5. Optionally, deploy the final model using the Flask app (app.py).

## Results & Evaluation

- Various models were compared on metrics such as precision, recall, F1-score, and ROC-AUC.
- Feature selection improved model accuracy and generalizability.
- Ensemble methods like Random Forest and Stacking showed superior performance.
- The final model can predict loan default risk with [insert your best metric here, e.g., 85% accuracy].

## Future Work

- Incorporate more granular borrower data and external economic indicators.
- Implement automated model retraining pipeline for production.
- Enhance web application UI for better user interaction.

## License

This project is licensed under the MIT License 
