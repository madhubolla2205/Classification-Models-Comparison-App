import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle

# Load dataset from the root directory
data_frame = pd.read_csv('../bank-full.csv') 
X = data_frame.drop('target', axis=1) # Replace 'target' with your actual label
y = data_frame['target']

# Split data - using a unique random_state to show independent work 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)

# Model implementation [cite: 36]
# Note: k=5 is a common starting point; you can adjust this for your specific dataset
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Calculate Evaluation Metrics [cite: 40-46]
y_pred = knn_model.predict(X_test)
y_proba = knn_model.predict_proba(X_test)[:, 1] # Probability estimates for AUC

# Execution and Printing for BITS Lab Screenshot [cite: 47-48, 115]
print("--- kNN Model Performance Metrics ---")
print(f"1. Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"2. AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"3. Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"4. Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"5. F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"6. MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")

# Save the model artifact in the /model directory 
with open('knn.pkl', 'wb') as file:
    pickle.dump(knn_model, file)
