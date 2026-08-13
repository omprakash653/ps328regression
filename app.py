# streamlit web development & cloud
import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.joblib')

#page configuration
st.set_page_config(page_title="package pred", page_icon=":guardsman", layout="centered")

#title
st.title("Package Prediction")
st.write("This is a simple web application that predicts the package based on the CGPA.")

#input CGPA
cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    try:
        #convert input to 2d array
        input_data = np.array([[cgpa]])
        #make prediction
        prediction = model.predict(input_data)
        st.success(f"The predicted package is: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
