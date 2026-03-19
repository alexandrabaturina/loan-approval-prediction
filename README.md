# Loan Approval Prediction
 
Predicting loan approval outcomes using logistic regression. The goal is to determine which applicants are likely to be approved for a loan based on their financial and personal attributes.
 
## Dataset
 
The dataset contains loan application records with the following features:
 
* `gender` — applicant's gender
* `married` — marital status
* `dependents` — number of dependents
* `education` — applicant's education level
* `self_employed` — whether the applicant is self-employed
* `applicant_income` — applicant's monthly income
* `coapplicant_income` — co-applicant's monthly income
* `loan_amount` — requested loan amount
* `loan_term` — loan term (in months)
* `credit_history` — credit history record
* `property_area` — area type of the property (Urban / Semiurban / Rural)
 
The target variable is `loan_status` (Y = Approved, N = Rejected).
 
## Project Structure
 
```
loan-approval-prediction/
│
├── LoanApprovalPrediction.csv        # Raw dataset
└── Loan Approval Prediction.ipynb    # Main notebook
```
 
## Methodology
 
1. **EDA** — distribution analysis, missing value inspection, visualization of feature relationships with loan status
2. **Preprocessing** — missing value imputation, categorical encoding, feature scaling inside a sklearn Pipeline, train/test split
3. **Baseline Model** — Logistic Regression (unbalanced classes)
4. **Class Imbalance Handling** — Logistic Regression with `class_weight='balanced'`
5. **Model Comparison** — Decision Tree Classifier evaluated via cross-validation
6. **Model Evaluation** — ROC-AUC, cross-validation, classification report, confusion matrix
 
## Results
 
| Model | Mean ROC-AUC (CV) | Min ROC-AUC (CV) | Accuracy |
| --- | --- | --- | --- |
| Logistic Regression (unbalanced) | 0.73 | 0.64 | 0.81 |
| Logistic Regression (balanced) | 0.73 | 0.65 | 0.81 |
| Decision Tree | 0.67 | 0.56 | — |
 
The **balanced Logistic Regression** was selected as the final model based on the best combination of mean and worst-case ROC-AUC across cross-validation folds.
 
## Key Findings
 
* `credit_history` is by far the strongest predictor of loan approval (coefficient 3.31)
* Living in a semiurban area, being married, and having a graduate degree increase approval odds
* A loan term of 36 months is the strongest negative predictor (coefficient -0.93), followed by a term of 480 months
* Having one dependent slightly decreases approval probability (coefficient -0.44)
* Class balancing improved the worst-case ROC-AUC from 0.64 to 0.65 with a marginal gain in mean ROC-AUC (0.7282 → 0.7290)
* The Decision Tree performed notably worse (mean ROC-AUC 0.67) and showed higher variance across folds
 
## Conclusion
 
The final model correctly predicted 96 out of 120 test cases: 75 approved loans (TP) and 21 rejected loans (TN). 

Among the 24 errors, 17 were false positives (high-risk borrowers mistakenly approved) and 7 were false negatives (creditworthy borrowers mistakenly rejected). 

In a credit scoring context, approving a bad borrower carries higher risk than rejecting a good one — making the balanced model a better fit despite rejecting more creditworthy applicants than the unbalanced version.
 
## Future Improvements
 
* **Hyperparameter tuning** — apply `GridSearchCV` or `RandomizedSearchCV` to optimize the `C` parameter (regularization strength) in Logistic Regression and `max_depth` in the Decision Tree
* **Additional models** — explore ensemble methods (Random Forest, XGBoost) to potentially improve predictive performance and reduce cross-validation variance
* **Feature engineering** — experiment with derived features such as total household income (`applicant_income + coapplicant_income`) or loan-to-income ratio
 
## Libraries
 
* `pandas`, `numpy` — data manipulation
* `scikit-learn` — preprocessing, pipeline, cross-validation, model evaluation
* `seaborn`, `matplotlib` — visualization
