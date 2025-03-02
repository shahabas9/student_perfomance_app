import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler


def load_model():
    with open("./student_model.pkl","rb") as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data["Extracurricular Activities"]=le.transform([data["Extracurricular Activities"]])[0]
    df=pd.DataFrame([data])
    scaled_data=scaler.transform(df)
    return scaled_data

def predict_data(data):
    model,scaler,le=load_model()
    preprocessed_data=preprocessing_input_data(data,scaler,le)
    prediction=model.predict(preprocessed_data)
    return prediction

def main():
    st.title("Student Perfomance Prediction")
    st.write("Enter your data for the prediction of student perfomance")

    hours=st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    prevoius_scores=st.number_input("Previous Scores",min_value=40,max_value=100,value=60)
    extra=st.selectbox("Extracurricular Activities",["Yes","No"])
    sleep=st.number_input("Sleep Hours",min_value=3,max_value=10,value=7)
    paper_solved=st.number_input("Question Paper Revised",min_value=1,max_value=10,value=3)

    if st.button("predict-your_score"):
        user_data={
            "Hours Studied" : hours,
            "Previous Scores" : prevoius_scores,
            "Extracurricular Activities" : extra,
            "Sleep Hours" :sleep,
            "Sample Question Papers Practiced" :paper_solved
        }
        prediction=predict_data(user_data)
        st.success(f"the perfomance is {prediction[0]}")

if __name__ == "__main__":
    main()