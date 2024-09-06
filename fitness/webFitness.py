from pycaret.regression import *
import streamlit as st
import pandas as pd
import numpy as np

# Load pretrained regression model
model = load_model('C:/Users/USER/Semester 4_MLOps/fitness/calories_regression_model')

# Define regression function to call
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predicted_calories = predictions_df['prediction_label'][0]
    return predicted_calories

def run():
    # Add title and subtitle to the main interface of the app
    st.title("Prediksi Kalori yang Terbakar")
    st.markdown("Aplikasi ini memprediksi jumlah kalori yang terbakar berdasarkan data aktivitas fisik")

    # Input form for user
    st.subheader("Masukkan Data Aktivitas Fisik:")
    steps = st.number_input('Jumlah Langkah', min_value=0, value=10000)
    total_distance = st.number_input('Total Jarak (km)', min_value=0.0, value=10.0)
    tracker_distance = st.number_input('Jarak yang Dicatat oleh Tracker (km)', min_value=0.0, value=8.0)
    logged_activities_distance = st.number_input('Jarak Aktivitas yang Dicatat (km)', min_value=0.0, value=2.0)
    very_active_distance = st.number_input('Jarak Aktivitas Sangat Aktif (km)', min_value=0.0, value=5.0)
    moderately_active_distance = st.number_input('Jarak Aktivitas Sedang (km)', min_value=0.0, value=3.0)
    light_active_distance = st.number_input('Jarak Aktivitas Ringan (km)', min_value=0.0, value=8.0)
    sedentary_active_distance = st.number_input('Jarak Aktivitas Tidak Aktif (km)', min_value=0.0, value=0.0)
    very_active_minutes = st.number_input('Menit Aktivitas Sangat Aktif', min_value=0, value=60)
    fairly_active_minutes = st.number_input('Menit Aktivitas Sedang', min_value=0, value=30)
    lightly_active_minutes = st.number_input('Menit Aktivitas Ringan', min_value=0, value=120)
    sedentary_minutes = st.number_input('Menit Aktivitas Tidak Aktif', min_value=0, value=360)
    
    # Create input dictionary
    input_dict = {
        'Steps': steps,
        'Total_Distance': total_distance,
        'Tracker_Distance': tracker_distance,
        'Logged_Activities_Distance': logged_activities_distance,
        'Very_Active_Distance': very_active_distance,
        'Moderately_Active_Distance': moderately_active_distance,
        'Light_Active_Distance': light_active_distance,
        'Sedentary_Active_Distance': sedentary_active_distance,
        'Very_Active_Minutes': very_active_minutes,
        'Fairly_Active_Minutes': fairly_active_minutes,
        'Lightly_Active_Minutes': lightly_active_minutes,
        'Sedentary_Minutes': sedentary_minutes
    }

    # Create DataFrame from input dictionary
    input_df = pd.DataFrame([input_dict])

    if st.button("Prediksi"):
        predicted_calories = predict(model=model, input_df=input_df)
        st.success(f"Jumlah Kalori yang Terbakar diprediksi: {predicted_calories:.2f} kkal")

if __name__ == '__main__':
    run()
