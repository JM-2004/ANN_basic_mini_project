import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('models/ann_model.keras')

with open('models/label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('models/one_hot_encoder_geography.pkl','rb') as file:
    ohe_geo=pickle.load(file)

with open('models/scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit app title
st.title("Customer Churn Prediction")
# Input fields for user data
geography = st.selectbox("Geography", ohe_geo.categories_[0])
gender= st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (in months)", value=5)
balance = st.number_input("Balance", value=5000.0)
credit_score = st.number_input("Credit Score", value=600)
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, value=1)
has_credit_card = {"Yes": 1, "No": 0}[st.selectbox("Has Credit Card", ["Yes", "No"])]
is_active_member = {"Yes": 1, "No": 0}[st.selectbox("Is Active Member", ["Yes", "No"])]


input_data={
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

input_data = pd.DataFrame(input_data)
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)
prediction_prob=prediction[0][0]
if(prediction_prob > 0.5):
    st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}.")
else:
    st.write(f"The customer is likely to stay. Probability of churn is {prediction_prob:.2f}.")