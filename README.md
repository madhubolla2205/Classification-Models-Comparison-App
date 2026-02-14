# Classification-Models-Comparison-App

## a. Problem Statement
The goal of this project is to build a **predictive classification system** for a retail banking institution to identify clients who are most likely to subscribe to a term deposit.

Direct marketing campaigns, specifically phone calls, are costly and time-consuming. Most clients contacted during these campaigns decline the offer, leading to inefficiencies. By leveraging historical customer data and machine learning, this project aims to:
* **Predict Outcomes:** Classify whether a client will subscribe (`yes`) or not (`no`) to a term deposit.
* **Model Comparison:** Evaluate and compare six different classification algorithms (**Logistic Regression, kNN, Naive Bayes, Decision Tree, Random Forest, and XGBoost**) to find the most robust predictor.
* **Optimization:** Use evaluation metrics like the **MCC (Matthews Correlation Coefficient)** to account for the inherent class imbalance (since most people do not subscribe), ensuring the bank targets the right audience and maximizes its marketing ROI.

---

## b. Dataset Description
The project utilizes the **Bank Marketing Dataset**, a well-known dataset from the UCI Machine Learning Repository related to direct marketing campaigns of a Portuguese banking institution.



### Dataset Overview
* **Total Records:** 45,211 rows.
* **Number of Features:** 16 independent variables (features) and 1 target variable.
* **Target Variable (y):** Binary classification (indicates if the client subscribed to a term deposit: "yes" or "no").

### Feature Categories
* **Bank Client Data:** Age, Job, Marital Status, Education, Default, Balance, Housing, and Personal Loans.
* **Last Contact Information:** Contact type, day, month, and duration of the last call.
* **Other Attributes:** Campaign frequency, pdays (days since last contact), and previous campaign outcomes.

---

## c. ML Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9031 | 0.9102 | 0.8902 | 0.9031 | 0.8897 | 0.4571 |
| **kNN** | 0.8897 | 0.8008 | 0.8705 | 0.8897 | 0.8737 | 0.3633 |
| **Naive Bayes** | 0.8552 | 0.8222 | 0.8751 | 0.8552 | 0.8636 | 0.4013 |
| **Decision Tree** | 0.9011 | 0.8535 | 0.8881 | 0.9011 | 0.8916 | 0.4301 |
| **Random Forest** | **0.9089** | **0.9345** | **0.8978** | **0.9089** | **0.8987** | 0.4894 |
| **XGBoost** | 0.9041 | 0.9304 | 0.8953 | 0.9041 | 0.8980 | **0.5052** |

---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed surprisingly well with an AUC of 0.9102, indicating a strong linear relationship between the encoded features and the target. |
| **kNN** | Showed the lowest AUC (0.8008) and MCC (0.3633). This suggests that high-dimensional bank data makes finding meaningful "neighbors" difficult, even with scaling. |
| **Naive Bayes** | Recorded the lowest accuracy (0.8552) but a decent MCC. This is likely due to the independence assumption being violated by correlated financial features. |
| **Decision Tree** | Provided good accuracy (0.9011) but a lower AUC than ensemble methods, as a single tree is prone to higher variance and less stable probability estimates. |
| **Random Forest** | Achieved the **highest AUC (0.9345)** and **Accuracy (0.9089)**, proving that bagging multiple trees effectively reduces overfitting on the bank dataset. |
| **XGBoost** | Delivered the **highest MCC (0.5052)**. Since MCC considers all quadrants of the confusion matrix, XGBoost is the most reliable model for handling the class imbalance in "yes" vs "no" subscriptions. |
