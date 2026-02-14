import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle


df = pd.read_csv('../bank-full.csv', sep=';')

df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop('y', axis=1)
y = df['y']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)

dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(X_train, y_train)


y_pred = dt_model.predict(X_test)
y_proba = dt_model.predict_proba(X_test)[:, 1]


print("--- Decision Tree Performance Metrics ---")
print(f"1. Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"2. AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"3. Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"4. Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"5. F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"6. MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")


with open('Decision_Tree.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
    
print("\nSuccess: Decision_Tree.pkl has been generated.")
