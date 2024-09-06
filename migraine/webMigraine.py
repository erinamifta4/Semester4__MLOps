from pycaret.classification import *
import streamlit as st
import pandas as pd

# Load pretrained classification model
model = load_model('C:/Users/USER/Semester 4_MLOps/migraine/migraine_model')

# Define classification function to call
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predicted_type = predictions_df['prediction_label'][0]
    return predicted_type

def run():
    # Add title and subtitle to the main interface of the app
    st.title("Prediksi Tipe Migrain")
    st.markdown("Aplikasi ini memprediksi tipe migrain berdasarkan data episode migrain")

    # Input form for user
    st.subheader("Masukkan Data Episode Migrain:")
    st.write("Durasi (jam): Durasi migrain dalam jam")
    st.write("Frekuensi: Jumlah gejala migrain dalam 1 bulan")
    st.write("Lokasi: Lokasi migrain dirasakan (0 untuk kepala bagian kanan, 1 untuk kepala bagian kiri)")
    st.write("Karakter: Karakteristik dari rasa sakit migrain")
    st.write("Intensitas: Intensitas migrain (jam)")
    st.write("Nausea: Kehadiran mual selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Muntah: Kehadiran muntah selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Phonophobia: Sensitivitas terhadap suara selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Photophobia: Sensitivitas terhadap cahaya selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Gangguan Visual: Gangguan visual selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Gangguan Sensorik: Gangguan sensorik selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Dysphasia: Kesulitan berbicara selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Dysarthria: Kesulitan berbicara jelas selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Vertigo: Sensasi pusing atau labil selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Tinnitus: Sensasi berdering atau berdengung di telinga selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Hypoacusis: Gangguan pendengaran selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Diplopia: Penglihatan ganda selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Defect: Defek atau gangguan visual selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Ataxia: Kesulitan koordinasi gerakan selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Conscience: Kehilangan kesadaran atau pingsan selama migrain (0 untuk tidak ada, 1 untuk ada)")
    st.write("Paresthesia: Sensasi kesemutan atau mati rasa selama migrain (0 untuk tidak ada, 1 untuk ada)")
    
    age = st.number_input('Usia', min_value=0, value=30)
    duration = st.number_input('Durasi (jam)', min_value=0, value=1)
    frequency = st.number_input('Frekuensi', min_value=0, value=5)
    location = st.number_input('Lokasi', min_value=0, value=1)
    character = st.number_input('Karakter', min_value=0, value=1)
    intensity = st.number_input('Intensitas', min_value=0, value=2)
    nausea = st.number_input('Nausea', min_value=0, value=1)
    vomit = st.number_input('Muntah', min_value=0, value=0)
    phonophobia = st.number_input('Phonophobia', min_value=0, value=1)
    photophobia = st.number_input('Photophobia', min_value=0, value=1)
    visual = st.number_input('Gangguan Visual', min_value=0, value=1)
    sensory = st.number_input('Gangguan Sensorik', min_value=0, value=2)
    dysphasia = st.number_input('Dysphasia', min_value=0, value=0)
    dysarthria = st.number_input('Dysarthria', min_value=0, value=0)
    vertigo = st.number_input('Vertigo', min_value=0, value=0)
    tinnitus = st.number_input('Tinnitus', min_value=0, value=0)
    hypoacusis = st.number_input('Hypoacusis', min_value=0, value=0)
    diplopia = st.number_input('Diplopia', min_value=0, value=0)
    defect = st.number_input('Defect', min_value=0, value=0)
    ataxia = st.number_input('Ataxia', min_value=0, value=0)
    conscience = st.number_input('Conscience', min_value=0, value=0)
    paresthesia = st.number_input('Paresthesia', min_value=0, value=0)
    
    # Create input dictionary
    input_dict = {
        'Age': age,
        'Duration': duration,
        'Frequency': frequency,
        'Location': location,
        'Character': character,
        'Intensity': intensity,
        'Nausea': nausea,
        'Vomit': vomit,
        'Phonophobia': phonophobia,
        'Photophobia': photophobia,
        'Visual': visual,
        'Sensory': sensory,
        'Dysphasia': dysphasia,
        'Dysarthria': dysarthria,
        'Vertigo': vertigo,
        'Tinnitus': tinnitus,
        'Hypoacusis': hypoacusis,
        'Diplopia': diplopia,
        'Defect': defect,
        'Ataxia': ataxia,
        'Conscience': conscience,
        'Paresthesia': paresthesia
    }

    # Create DataFrame from input dictionary
    input_df = pd.DataFrame([input_dict])

    if st.button("Prediksi"):
        predicted_type = predict(model=model, input_df=input_df)
        st.success(f"Tipe migrain yang diprediksi: {predicted_type}")

if __name__ == '__main__':
    run()
