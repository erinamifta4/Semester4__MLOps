import streamlit as st
import pandas as pd
from pycaret.classification import *

# Load pretrained classification model
model = load_model('C:/Users/USER/Semester 4_MLOps/kelayakan_ukt/kelayakan_ukt')

# Define classification function to call
def predict(model, input_df):
    # Predict label with the model
    predictions_df = predict_model(model, data=input_df)
    
    # Get the predicted label
    predicted_label = predictions_df['prediction_label'][0]
    return predicted_label

def load_data():
    return pd.read_csv('C:/Users/USER/Semester 4_MLOps/kelayakan_ukt/klasifikasi_mahasiswa.csv')

def run():
    # Add title and subtitle to the main interface of the app
    st.title("Prediksi Kelayakan Keringanan UKT")
    st.markdown("<style>h1{color: #1E302C;}</style>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Uang Kuliah Tunggal (UKT)")
    st.sidebar.image("C:/Users/USER/Semester 4_MLOps/kelayakan_ukt/ukt2.png", use_column_width=True)

    # Menu navigasi
    menu = st.sidebar.radio("Menu", ["Prediksi Kelayakan", "Grafik Data", "Informasi"])

    if menu == "Prediksi Kelayakan":
        # Load data from CSV
        df = load_data()

        # Input form for user
        st.subheader("Masukkan Data Mahasiswa:")
        nama = st.text_input('Nama')
        nrp = st.text_input('NRP')
        tempat_tinggal_mapping = {"Perdesaan": 0, "Perkotaan": 1}  # mapping dictionary
        tempat_tinggal = st.selectbox('Tempat Tinggal', list(tempat_tinggal_mapping.keys()))

        pekerjaan_ortu = st.selectbox('Pekerjaan Orang Tua', df['Pekerjaan Orang Tua'].unique())
        penghasilan_ortu = st.number_input('Penghasilan Orang Tua', min_value=0, value=2000000, step=50000)
        jumlah_tanggungan = st.number_input('Jumlah Tanggungan Orang Tua', min_value=0, value=4)
        kendaraan = st.selectbox('Kendaraan Pribadi', df['Kendaraan'].unique())

        # Create input dictionary
        input_dict = {
            'Nama': nama,
            'NRP': nrp,
            'Tempat Tinggal': tempat_tinggal_mapping[tempat_tinggal],  # map user input to corresponding value
            'Pekerjaan Orang Tua': pekerjaan_ortu,
            'Penghasilan Orang Tua': penghasilan_ortu,
            'Jumlah Tanggungan Orang Tua': jumlah_tanggungan,
            'Kendaraan': kendaraan
        }

        # Create DataFrame from input dictionary
        input_df = pd.DataFrame([input_dict])

        if st.button("Prediksi"):
            predicted_label = predict(model=model, input_df=input_df)
            if predicted_label == 0:
                st.error(f"Maaf, {nama} dengan NRP {nrp} tidak memenuhi syarat untuk mendapatkan keringanan UKT.")
            else:
                st.success(f"Selamat! {nama} dengan NRP {nrp} memenuhi syarat untuk mendapatkan keringanan UKT. Silahkan lengkapi data administrasi untuk tahap berikutnya!")


    elif menu == "Grafik Data":
        # Load data
        df = load_data()

        # Plotting charts based on data
        st.subheader("Grafik Kelayakan UKT Berdasarkan Tempat Tinggal")
        st.bar_chart(df.groupby('Tempat Tinggal')['Kelayakan Keringanan UKT'].mean(), color=['#4D351D'])
        st.text("Tempat Tinggal")

        st.subheader("Grafik Kelayakan UKT Berdasarkan Pekerjaan Orang Tua")
        st.bar_chart(df.groupby('Pekerjaan Orang Tua')['Kelayakan Keringanan UKT'].mean(), color=['#1E302C'])
        st.text("Pekerjaan Orang Tua")

        st.subheader("Grafik Kelayakan UKT Berdasarkan Jumlah Tanggungan Orang Tua")
        st.line_chart(df.groupby('Jumlah Tanggungan Orang Tua')['Kelayakan Keringanan UKT'].mean(), color=['#4D351D'])
        st.text("Jumlah Tanggungan Orang Tua")

        st.subheader("Grafik Kelayakan UKT Berdasarkan Kendaraan Pribadi")
        st.bar_chart(df.groupby('Kendaraan')['Kelayakan Keringanan UKT'].mean(), color=['#1E302C'])
        st.text("Kendaraan Pribadi")

    else:  # Informasi
        #st.image("C:/Users/USER/Semester 4_MLOps/kelayakan_ukt/logopens.png")
        st.title("Tentang Keringanan UKT")
        st.markdown("Keringanan Uang Kuliah Tunggal (UKT) adalah sebuah sistem yang memberikan bantuan keuangan kepada mahasiswa yang kurang mampu secara ekonomi untuk membiayai pendidikan selama di perguruan tinggi.")
        st.markdown("Sistem ini bertujuan untuk memastikan akses pendidikan yang lebih adil dan merata bagi semua lapisan masyarakat, sehingga memungkinkan lebih banyak individu untuk mengejar pendidikan tinggi.")

        st.title("Tentang Aplikasi")
        st.markdown("Ini adalah aplikasi sederhana untuk memprediksi kelayakan mahasiswa untuk mendapatkan keringanan Uang Kuliah Tunggal (UKT) di Politeknik Elektronika Negeri Surabaya.")
        st.markdown("Dibuat oleh Erina Nur Miftaqul Jannah, mahasiswa Program Studi Sarjana Terapan [Sains Data Terapan](https://sainsdata.pens.ac.id), [Politeknik Elektronika Negeri Surabaya](http://pens.ac.id).")

    st.sidebar.markdown("© 2024 Erina NMJ - PENS")


if __name__ == '__main__':
    run()
