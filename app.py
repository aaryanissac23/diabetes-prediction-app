import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("🩺 Diabetes Prediction System")

preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ Person is Diabetic")
    else:
        st.success("✅ Person is NOT Diabetic")
