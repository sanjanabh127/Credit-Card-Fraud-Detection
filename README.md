# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas) 
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy) 


A **Machine Learning–based Credit Card Fraud Detection System** that identifies fraudulent transactions using supervised learning models and deploys predictions through a **Flask web application**.

---

##  Project Overview

Credit card fraud is a major financial threat in digital transactions.  
This project uses **machine learning classification algorithms** to detect fraudulent transactions based on transaction features and patterns.

The trained model is integrated into a **Flask web app** that allows users to input transaction details and receive instant fraud predictions.

---

##  Technologies Used

- **Programming Language:** Python  
- **Machine Learning:** scikit-learn  
- **Web Framework:** Streamlit
- **Data Handling:** Pandas, NumPy  
- **Visualization & Analysis:** Jupyter Notebook  
- **Version Control:** Git & GitHub  

---

##  Dataset

- **Source:** Kaggle – Credit Card Fraud Detection Dataset  
- **Link:** https://www.kaggle.com/mlg-ulb/creditcardfraud  

> ⚠️ The dataset is not included in this repository due to GitHub file size limits.

---

##  Workflow 

1. **Data Collection**  
   - Collected anonymized credit card transaction data from Kaggle.

2. **Data Preprocessing**  
   - Scaled numerical features and handled class imbalance.
   - Split the dataset into training and testing sets.

3. **Model Training**  
   - Trained a supervised machine learning classification model using scikit-learn.

4. **Model Evaluation**  
   - Evaluated model performance to ensure accurate fraud detection.

5. **Model Deployment**  
   - Saved the trained model using `joblib`.
   - Integrated the model into a Flask web application for real-time predictions.

---

###  Results

- Successfully classifies transactions as **Fraudulent** or **Legitimate**.
- Provides instant predictions via the web interface.
- Demonstrates the effectiveness of machine learning in financial fraud detection.

---

###   Output

- **Legitimate Transaction:** Transaction is safe  
- **Fraudulent Transaction:** Fraud alert detected  
  ![img_alt](https://github.com/sanjanabh127/Credit-Card-Fraud-Detection/blob/6942944c1dd19b43aa9002703ff37eac3442ec62/Screenshots/Screenshot%202025-12-24%20185031.png)
  ![img_alt](https://github.com/sanjanabh127/Credit-Card-Fraud-Detection/blob/6942944c1dd19b43aa9002703ff37eac3442ec62/Screenshots/Screenshot%202025-12-24%20185444.png)
---

###  Key Takeaway

This project demonstrates:
- End-to-end machine learning workflow
- Real-world fraud detection use case
- Integration of ML models with web applications
- Practical deployment of data science solutions

---



