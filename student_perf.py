import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi




uri = "mongodb+srv://shabunfiz:7gX2bN72xjVcuVZv@shahabas.b0u5f.mongodb.net/?retryWrites=true&w=majority&appName=shahabas"
# Create a new client and connect to the server
client = MongoClient(uri)
db=client['Student']
collection=db["perfomance"]


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
        user_data = {
            "Hours Studied": int(hours),  # Ensure conversion to Python int
            "Previous Scores": int(prevoius_scores),
            "Extracurricular Activities": str(extra),
            "Sleep Hours": int(sleep),
            "Sample Question Papers Practiced": int(paper_solved)
        }
        prediction = predict_data(user_data)
        st.success(f"The performance is {prediction[0]}")
        
        user_data["prediction"] = float(prediction[0])  # Convert prediction to float if needed
        
        for key, value in user_data.items():
            if isinstance(value, (np.int64, np.float64)):
                print(key,value)
                user_data[key] = value.item()  # Converts numpy types to native Python
        
        collection.insert_one(user_data)

if __name__ == "__main__":
    main()