# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:28:13 2024

@author: Hp
"""

import numpy as np
import pickle
import streamlit as st

# Load the model
loaded_model = pickle.load(open("D:/project/diabetes prediction/trained_model.sav", 'rb'))


# Create function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert input to numpy array
    input_data_as_array = np.asarray(input_data, dtype=float)
    
    # Reshape array as we are predicting for one instance
    input_reshape = input_data_as_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_reshape)

    if prediction[0] == 0:
        return "Person is not Diabetic"
    else:
        return "Person is Diabetic"

# Main function for Streamlit app
def main():
    # Give title
    st.title('Diabetes Prediction App')
   
    
    # Get user input
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age of the Person")

    # Code for prediction
    diagnosis = ''
    
    # Create a button for prediction
    if st.button("Diabetes Test Result"):
        try:
            # Convert inputs to float
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "Please enter valid numerical values."
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()

