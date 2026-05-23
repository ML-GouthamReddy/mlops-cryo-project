import streamlit as st
import pickle
import numpy as np

# Load the trained model artifact
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Cryotherapy Treatment Prediction")
st.write("Enter the patient details below to predict treatment success.")

sex = st.selectbox("Sex", [1, 2])
age = st.number_input("Age", min_value=1, max_value=100, value=35)
time = st.number_input("Time", min_value=0.0, value=12.0)
number_of_warts = st.number_input("Number of Warts", min_value=1, value=5)
type_of_wart = st.selectbox("Type", [1, 2, 3])
area = st.number_input("Area", min_value=0, value=100)

if st.button("Predict"):
    features = np.array([[sex, age, time, number_of_warts, type_of_wart, area]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("Result: Success (1)")
    else:
        st.error("Result: Failure (0)")