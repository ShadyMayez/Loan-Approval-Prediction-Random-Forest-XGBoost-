# Loan Approval Prediction (97% Accuracy)

## Project Overview and Purpose
This project implements a binary classification system to automate the loan approval process. By analyzing features such as credit score, income, and debt-to-income ratios, the model predicts whether a loan application should be approved or rejected. The project focuses on handling class imbalance and optimizing ensemble models to achieve high precision and recall in a financial context.

## Key Technologies and Libraries
- **Language**: Python
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `catboost`
- **Imbalance Handling**: Oversampling techniques for minority classes.

## Methodology and Workflow
### Data Preprocessing
1. **Categorical Encoding**: Implemented a custom automated encoder that identifies binary categories for `LabelEncoding` and multi-class categories for `OneHotEncoding`.
2. **Handling Imbalance**: Addressed the skew in the dataset (77% majority vs 22% minority) by oversampling the minority class to a 50:50 distribution (70,000 samples).
3. **Feature Engineering**: Normalized numerical inputs like `person_income` and `loan_amnt` to improve model convergence.

### Model Performance
The project evaluates several high-performance classifiers:
- **Random Forest Classifier**: Achieved the top accuracy of **~97%** on the balanced dataset.
- **XGBoost & CatBoost**: Used as comparative benchmarks for gradient boosting performance.


## Results and Insights
- **Key Predictors**: Credit score and the percentage of income dedicated to the loan were identified as the most significant predictors of approval.
- **Evaluation Metrics**: Beyond simple accuracy, the project includes detailed confusion matrices and F1-scores to ensure the model performs reliably on both approved and rejected cases.

## How to Run
1. **Dataset**: Ensure the `loan_data.csv` is present in the working directory.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
