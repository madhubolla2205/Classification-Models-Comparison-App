import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle

# Load dataset from the root directory [cite: 51-55]
# Replace 'your_dataset.csv' with your actual filename
final_data = pd.read_csv('../bank-full.csv') 
X = final_data.drop('target', axis=1) # Replace 'target' with your actual label
y = final_data['target']

# Split data - using a unique random_state for academic integrity [cite: 98-100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=106)

# Ensemble Model implementation 
# use_label_encoder=False and eval_metric='logloss' prevent common warnings
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_classifier.fit(X_train, y_train)

# Calculate Evaluation Metrics [cite: 40-46]
y_pred = xgb_classifier.predict(X_test)
y_proba = xgb_classifier.predict_proba(X_test)[:, 1] 

# Execution and Printing for BITS Lab Screenshot [cite: 22, 47-48, 115]
print("--- XGBoost Performance Metrics ---")
print(f"1. Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"2. AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"3. Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"4. Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"5. F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"6. MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")

# Save the model artifact in the /model directory
with open('xgboost.pkl', 'wb') as output_file:
    pickle.dump(xgb_classifier, output_file)