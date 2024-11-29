# Loan Default Prediction Project

## Overview
This project aims to predict whether a loan will be **fully paid** or **not fully paid** based on various features such as credit policy adherence, FICO score, interest rates, and loan purpose. The dataset used is `loan_data.csv`, and multiple machine learning models are implemented and evaluated to determine their performance.

## Key Features
- **Dataset Preprocessing**:
  - Encoding categorical features.
  - Checking and handling missing values.
  - Class distribution analysis for `not.fully.paid`.

- **Data Visualization**:
  - Histograms for FICO scores by `credit.policy` and loan status.
  - Count plots for loan purpose vs. `not.fully.paid`.
  - Heatmap showing feature correlations.
  - Linear regression plot for FICO vs. interest rate.

- **Machine Learning Models**:
  - **Decision Tree Classifier**: Baseline model with limited depth.
  - **Bagging Classifier**: Combines multiple decision trees to reduce variance.
  - **AdaBoost Classifier**: Boosts weak learners for better performance.
  - **Random Forest Classifier**: Aggregates multiple decision trees to improve accuracy.
  - **Gradient Boosting Classifier**: Combines models sequentially to reduce errors.

- **Evaluation Metrics**:
  - Confusion Matrix.
  - Precision, Recall, and F1-Score.
  - Accuracy on test set and cross-validation scores.

## Results Summary
| **Model**                  | **Test Accuracy** | **Cross-Validation Accuracy** |
|----------------------------|-------------------|--------------------------------|
| Decision Tree              | 84.59%           | 73.10%                         |
| Bagging with Decision Tree | 84.59%           | 73.10%                         |
| AdaBoost                   | 84.38%           | 73.10%                         |
| Random Forest              | 84.76%           | 72.99%                         |
| Gradient Boosting          | 84.41%           | 73.08%                         |

**Key Insight**: All models performed similarly in terms of accuracy but struggled with the imbalanced nature of the dataset, as evidenced by the low recall for the minority class (`not.fully.paid = 1`).

## Technologies Used
- **Languages**: Python
- **Libraries**: 
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
- **Tools**: Jupyter Notebook, IDE (PyCharm/VSCode)

## Challenges
- **Class Imbalance**: The dataset has a skewed distribution (majority class: `0`, minority class: `1`), leading to poor recall for the minority class.
- **Model Tuning**: Selecting appropriate hyperparameters for ensemble models to balance bias and variance.

## Future Work
- **Class Balancing**: Apply techniques such as SMOTE or cost-sensitive learning to address class imbalance.
- **Feature Engineering**: Create additional meaningful features to improve model performance.
- **Hyperparameter Optimization**: Use grid search or randomized search for better tuning of ensemble models.

This project demonstrates practical implementation of machine learning models, data visualization, and handling imbalanced datasets in the domain of finance. Feedback and suggestions are welcome!
