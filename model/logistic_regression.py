import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle

# Load dataset from the root directory [cite: 51-55]
data = pd.read_csv('../bank-full.csv') 
X = data.drop('target', axis=1) # Replace 'target' with your actual label column
y = data['target']

# Split data - using a unique random_state for variety [cite: 98-100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Model implementation [cite: 34]
# Increased max_iter to ensure convergence on complex datasets
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, y_train)

# Calculate Evaluation Metrics [cite: 40-46]
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1] 

# Execution and Printing for BITS Lab Screenshot 
print("--- Logistic Regression Performance Metrics ---")
print(f"1. Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"2. AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"3. Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"4. Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"5. F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"6. MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")

# Save the model artifact in the /model directory 
with open('logistic_regression.pkl', 'wb') as model_file:
    pickle.dump(log_reg, model_file)