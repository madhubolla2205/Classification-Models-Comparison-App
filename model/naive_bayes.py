import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import pickle

raw_data = pd.read_csv('../bank-full.csv', sep=';') 

raw_data['y'] = raw_data['y'].map({'yes': 1, 'no': 0})

X = raw_data.drop('y', axis=1)
y = raw_data['y']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
y_proba = nb_classifier.predict_proba(X_test)[:, 1] 

print("--- Naive Bayes Performance Metrics ---")
print(f"1. Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"2. AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"3. Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"4. Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"5. F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"6. MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")

with open('Naive_Bayes.pkl', 'wb') as f_out:
    pickle.dump(nb_classifier, f_out)

print("\nSuccess: Naive_Bayes.pkl generated.")