import streamlit as st
import joblib
import pandas as pd

st.title('Sales Prediction App')
st.write('Predict sales based on advertising budgets for TV, Radio, and Newspaper.')

# Load the trained model
try:
    model = joblib.load('linear_regression_model.sav')
    st.success('Model loaded successfully!')
except FileNotFoundError:
    st.error('Error: linear_regression_model.sav not found. Please ensure the model is saved in the same directory.')
    st.stop()

# Input fields for features
st.sidebar.header('Input Advertising Budgets')
tv = st.sidebar.slider('TV Budget (in thousands)', 0.0, 300.0, 150.0)
radio = st.sidebar.slider('Radio Budget (in thousands)', 0.0, 50.0, 25.0)
newspaper = st.sidebar.slider('Newspaper Budget (in thousands)', 0.0, 120.0, 40.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'TV': tv,
    'Radio': radio,
    'Newspaper': newspaper
}])

st.subheader('Input Data')
st.write(input_data)

# Make prediction
if st.button('Predict Sales'):
    prediction = model.predict(input_data)[0]
    st.subheader('Predicted Sales')
    st.write(f'The predicted sales are: **{prediction:.2f}** (in thousands)')
