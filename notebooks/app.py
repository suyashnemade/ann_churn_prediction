import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


##loading trained model
model = tf.keras.models.load_model('model1.h5')

#encoders
with open('label_encoder_gender.pkl', 'rb') as f:
    encoder_gender = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    encoder_geography = pickle.load(f)

with open('sc.pkl', 'rb') as f:
    scaler = pickle.load(f)


#streamlit app
st.title('Customer Churn Prediction')
st.write('Predict whether a customer will churn or not')

#input fields
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=25)
balance = st.number_input('Balance', min_value=0, max_value=200000, value=50000)
est_salary = st.number_input('Estimated Salary', min_value=0, max_value=200000, value=50000)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
no_of_products = st.number_input('Number of Products', min_value=1, max_value=10, value=1)
has_credit_card = st.selectbox('Has Credit Card', [1, 0])
is_active_member = st.selectbox('Is Active Member', [1, 0])


#input

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [est_salary]
})

#OHE GEOGRAPHY
geo_encode = encoder_geography.transform([[geography]]).toarray()
geo_encoded_tf = pd.DataFrame(geo_encode, columns=encoder_geography.get_feature_names_out(['Geography']))

#combine ohe and encoded column

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_tf], axis=1)


#scale
input_data_scaled = scaler.transform(input_data)

#predict
prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

#output
if pred_prob > 0.5:
    st.write(f'Customer will churn with probability {pred_prob:.2f}')
else:
    st.write(f'Customer will not churn with probability {1-pred_prob:.2f}')