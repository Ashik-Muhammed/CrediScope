# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reading the dataset
df = pd.read_csv("loan_data.csv")

# Check for missing values
if df.isnull().sum().sum() > 0:
    print("Missing values in the dataset:", df.isnull().sum())

# Encoding 'purpose' column
df['purpose'] = LabelEncoder().fit_transform(df['purpose'])

# Check class distribution
print("Class distribution:\n", df['not.fully.paid'].value_counts())

# Data visualization
sns.set_style('darkgrid')

# FICO score distribution based on credit policy
plt.figure(figsize=(10, 6))
plt.hist(df['fico'].loc[df['credit.policy'] == 1], bins=30, alpha=0.5, label='Credit.Policy=1')
plt.hist(df['fico'].loc[df['credit.policy'] == 0], bins=30, alpha=0.5, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO Score')
plt.title('FICO Score Distribution by Credit Policy')
plt.show()

# FICO score distribution based on loan status
plt.figure(figsize=(10, 6))
df[df['not.fully.paid'] == 1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='not.fully.paid=1')
df[df['not.fully.paid'] == 0]['fico'].hist(bins=30, alpha=0.5, color='green', label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO Score')
plt.title('FICO Score Distribution by Loan Status')
plt.show()

# Count plot of loan purpose vs loan status
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.title('Loan Purpose vs Not Fully Paid')
plt.show()

# Linear model plot for FICO vs interest rate colored by credit policy and loan status
sns.lmplot(data=df, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set2')
plt.title('FICO vs Interest Rate by Credit Policy and Loan Status')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), cmap='BuPu', annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Train-Test Split
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(max_depth=2)
dt_clf.fit(X_train, y_train)
y_pred_test = dt_clf.predict(X_test)

print("<-- Decision Tree -->")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print("Test Accuracy: ", accuracy_score(y_test, y_pred_test))

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt_clf, X, y, cv=StratifiedKFold(n_splits=5))
print("Cross-Validated Accuracy for Decision Tree: ", cv_scores_dt.mean())

# Bagging with Decision Tree
bag_dt = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, bootstrap=True)
bag_dt.fit(X_train, y_train)
y_pred_bag = bag_dt.predict(X_test)

print("<-- Bagging with Decision Tree -->")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bag))
print(classification_report(y_test, y_pred_bag))
print("Test Accuracy: ", accuracy_score(y_test, y_pred_bag))

# Cross-validation for Bagging Classifier
cv_scores_bag = cross_val_score(bag_dt, X, y, cv=StratifiedKFold(n_splits=5))
print("Cross-Validated Accuracy for Bagging Classifier: ", cv_scores_bag.mean())

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME')  # Updated to avoid future warnings
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

print("<-- AdaBoost Classifier -->")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))
print("Test Accuracy: ", accuracy_score(y_test, y_pred_ada))

# Cross-validation for AdaBoost Classifier
cv_scores_ada = cross_val_score(ada_clf, X, y, cv=StratifiedKFold(n_splits=5))
print("Cross-Validated Accuracy for AdaBoost Classifier: ", cv_scores_ada.mean())

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("<-- Random Forest Classifier -->")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Test Accuracy: ", accuracy_score(y_test, y_pred_rf))

# Cross-validation for Random Forest Classifier
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=StratifiedKFold(n_splits=5))
print("Cross-Validated Accuracy for Random Forest Classifier: ", cv_scores_rf.mean())

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)

print("<-- Gradient Boosting Classifier -->")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
print("Test Accuracy: ", accuracy_score(y_test, y_pred_gb))

# Cross-validation for Gradient Boosting Classifier
cv_scores_gb = cross_val_score(gb_clf, X, y, cv=StratifiedKFold(n_splits=5))
print("Cross-Validated Accuracy for Gradient Boosting Classifier: ", cv_scores_gb.mean())
