from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np

#load pretrained model for classification
model = load_model('C:/Users/USER/Semester 4_MLOps/diabetes/diabetes_pipeline')

#define classification function to call
def predict(model, input_df):
    predictions_df = predict_model(model, data = input_df)
    st.write(predictions_df)
    predictions= predictions_df['prediction_label'][0]
    return predictions

def run():
    #mengambil gambar yang ada difolder
    from PIL import Image
    image = Image.open('C:/Users/USER/Semester 4_MLOps/diabetes/images/logo_sdt.png')
    image_diabetes = Image.open('C:/Users/USER/Semester 4_MLOps/diabetes/images/diabetes.jpg')

    #add sidebar to the app
    st.sidebar.title('Praktikum Streamlit')
    st.sidebar.markdown("Aplikasi klasifikasi diabetes berdasarkan beberapa fitur dari data yang disediakan oleh pycaret")
    st.sidebar.info("Aplikasi ini contoh praktikum streamlit pada mata kuliah MLOps")
    st.sidebar.success("By: Renovita Edelani")
    st.sidebar.image(image)
    
    #add title and subtitle to the main interface of the app
    st.image(image_diabetes)
    st.title("Klasifikasi Penyakit Diabetes")
    st.markdown("Diabetes merupakan suatu kondisi medis yang ditandai dengan kadar glukosa (gula) darah yang tinggi dalam jangka waktu yang lama.")
    pregnant = st.number_input('Number of times pregnant', min_value=0, max_value=20, value=2)
    plasma = st.number_input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test',min_value=0, max_value=300, value=90)
    bp = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0, max_value=200, value=80)
    ts = st.number_input('Triceps skin fold thickness (mm)', min_value=0, max_value=150, value=0)
    serum = st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0, max_value=900, value=0)
    bmi = st.number_input('Body mass index (weight in kg/(height in m)^2)', min_value=0.0, max_value=80.0, value=32.0, step=0.1)
    dp = st.number_input('Diabetes pedigree function',min_value=0.000, max_value=3.000, value=0.258, step=0.001, format="%.3f")
    age = st.number_input('Age (years)',min_value=0, max_value=100, value=22)
    
    input_dict = {'Number of times pregnant' : pregnant,
                 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test' : plasma,
                 'Diastolic blood pressure (mm Hg)' : bp,
                  'Triceps skin fold thickness (mm)' :ts ,
                  '2-Hour serum insulin (mu U/ml)' : serum,
                  'Body mass index (weight in kg/(height in m)^2)' : bmi,
                  'Diabetes pedigree function' : dp,
                  'Age (years)' : age,
                 }
    input_df = pd.DataFrame([input_dict])
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        if output:
            st.error('Terkena Diabetes')
        else:
            st.success('Tidak terkena Diabetes')

   
if __name__ == '__main__':
    run()