import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

st.title("Bank Marketing Classification Dashboard")

# a. Dataset upload option (CSV)
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'bank-full.csv' or similar test file", type="csv")

# b. Model selection dropdown
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox("Choose Model", 
    ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"])

if uploaded_file:
    # Read data - Bank dataset uses ';' as a separator
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("### Test Data Preview")
    st.dataframe(df.head())

    if st.button("Run Evaluation"):
        try:
            # 1. Preprocessing (Must match your training scripts!)
            # Convert categorical text to dummies
            df_processed = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
            
            # Map target 'y' to 1/0
            if 'y' in df_processed.columns:
                df_processed['y'] = df_processed['y'].map({'yes': 1, 'no': 0})
                X_test = df_processed.drop('y', axis=1)
                y_test = df_processed['y']
            else:
                st.error("Target column 'y' not found in CSV.")
                st.stop()

            # 2. Load Model
            with open(f'model/{model_option}.pkl', 'rb') as f:
                model = pickle.load(f)

            # 3. Predict
            y_pred = model.predict(X_test)

            # c. Display Metrics
            st.write(f"## {model_option} Performance")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            col2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

            # d. Confusion Matrix Visualization
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

            st.write("### Detailed Report")
            st.text(classification_report(y_test, y_pred))

        except FileNotFoundError:
            st.error(f"Error: model/{model_option}.pkl not found. Please run your training scripts first.")
        except Exception as e:
            st.error(f"An error occurred: {e}")