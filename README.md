# Customer Churn Prediction App

This is a Streamlit-based web application that predicts whether a customer is likely to churn based on input features such as age, balance, geography, credit score, etc.

## 🧠 Model

The app uses a trained Artificial Neural Network (ANN) model built with TensorFlow/Keras, along with preprocessing tools such as:
- `StandardScaler` for feature scaling
- `LabelEncoder` for gender
- `OneHotEncoder` for geography


```bash
pip install streamlit pandas numpy scikit-learn tensorflow
