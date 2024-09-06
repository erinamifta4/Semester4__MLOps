from pycaret.classification import *
import streamlit as st
import pandas as pd

# Load model klasifikasi yang sudah disimpan sebelumnya
model = load_model('C:/Users/USER/Semester 4_MLOps/heart_failure/heart_failure_classification_model')

# Fungsi untuk melakukan prediksi menggunakan model
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():
    # Judul dan deskripsi aplikasi
    st.title("Klasifikasi Kematian Pasien dengan Penyakit Jantung")
    st.markdown("Aplikasi ini dapat memprediksi apakah seorang pasien dengan penyakit jantung akan meninggal selama periode follow-up.")

    # Memasukkan nilai fitur dari pengguna
    age = st.number_input('Usia (tahun)', min_value=0, max_value=120, value=50)
    anaemia = st.selectbox('Anaemia (0 = Tidak, 1 = Ya)', [0, 1])
    cpk = st.number_input('Creatinine phosphokinase (mcg/L)', min_value=0, value=100)
    diabetes = st.selectbox('Diabetes (0 = Tidak, 1 = Ya)', [0, 1])
    ejection_fraction = st.number_input('Ejection fraction (%)', min_value=0, max_value=100, value=50)
    high_blood_pressure = st.selectbox('High blood pressure (0 = Tidak, 1 = Ya)', [0, 1])
    platelets = st.number_input('Platelets (kiloplatelets/mL)', min_value=0, value=200000)
    serum_creatinine = st.number_input('Serum creatinine (mg/dL)', min_value=0.0, value=1.0, step=0.1)
    serum_sodium = st.number_input('Serum sodium (mEq/L)', min_value=0, value=135)
    sex = st.selectbox('Sex (0 = Perempuan, 1 = Laki-laki)', [0, 1])
    smoking = st.selectbox('Smoking (0 = Tidak, 1 = Ya)', [0, 1])
    time = st.number_input('Waktu follow-up (hari)', min_value=0, value=150)

    # Membuat dataframe dari input pengguna
    input_dict = {'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': cpk, 'diabetes': diabetes,
                  'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure,
                  'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
                  'sex': sex, 'smoking': smoking, 'time': time}
    input_df = pd.DataFrame([input_dict])

    # Tombol prediksi
    if st.button("Prediksi"):
        output = predict(model=model, input_df=input_df)
        if output == 1:
            st.error('Pasien kemungkinan akan meninggal selama periode follow-up.')
        else:
            st.success('Pasien kemungkinan tidak akan meninggal selama periode follow-up.')

if __name__ == '__main__':
    run()
