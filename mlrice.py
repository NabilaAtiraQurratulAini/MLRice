import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.title("""DATA RICE CAMMEO AND OSMANCIK""")
st.write("### KELOMPOK 14")
st.write("Nama Anggota :")
st.write("1. Nabila Atira Qurratul Aini / 210411100066")
st.write("2. Firdatul A'yuni / 210411100144")
st.write("3. Jennatul Macwe / 210411100151")

# mengasumsikan 'st' adalah instansi Streamlit
tabs = st.tabs(["Dataset Description", "Data File", "Preprocessing", "Random Forest", "Inputan Data"])
dataset_description, data_file, preprocessing, rf, inputan = tabs

with dataset_description:
    st.write("### DATASET")
    st.write("### BUSINESS UNDESTANDING - TUJUAN")
    st.write("Tujuan dari penelitian ini adalah untuk meningkatkan pemahaman dan klasifikasi terhadap dua spesies padi bersertifikat di Turki, yaitu Osmancik dan Cammeo. Dengan memiliki pemahaman yang lebih baik tentang karakteristik dan perbedaan antara kedua spesies ini, kita dapat mendukung pengembangan kebijakan pertanian yang lebih efektif dan meningkatkan produktivitas tanaman padi bersertifikat di Turki.")
    st.write("### DATA UNDERSTANDING - INFORMASI DATA DAN FITUR")
    st.write("- Area atau Daerah : Fitur ini mengukur luas dari objek. Luas dapat dihitung dalam unit piksel atau unit luas lainnya, tergantung pada resolusi data. Luas memberikan informasi tentang ukuran relatif objek. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Perimeter : Perimeter mengukur panjang garis batas objek. Ini diukur sebagai jumlah panjang semua tepi objek. Perimeter bisa memberikan indikasi seberapa kompleks bentuk objek tersebut. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Major Axis Length atau Panjang Sumbu Utama : Sumbu utama adalah sumbu terpanjang dalam elips yang mengelilingi objek. Panjang sumbu ini memberikan gambaran tentang dimensi utama objek dan arah orientasi elips. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Minor Axis Length atau Panjang Sumbu Kecil : Sumbu kecil adalah sumbu terpendek dalam elips yang mengelilingi objek. Ini memberikan informasi tentang dimensi kedua objek dan dapat membantu menggambarkan bentuk elips. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Eccentricity atau Eksentrisitas : Eksentrisitas mengukur sejauh mana elips yang mengelilingi objek mendekati bentuk lingkaran. Nilai eksentrisitas 0 menunjukkan objek yang bentuknya mendekati lingkaran sempurna, sementara nilai mendekati 1 menunjukkan elips yang sangat panjang. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Convex Area atau Daerah Cembung : Luas cembung mengukur luas daerah yang diukur dari cembung (convex hull) objek. Cembung adalah poligon terkecil yang dapat mencakup seluruh objek. Luas ini memberikan informasi tentang sejauh mana objek dapat dianggap 'cembung'. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Extent atau Luas : Extent adalah rasio antara luas objek dan luas kotak terkecil yang dapat mengelilingi objek. Nilai 1 menunjukkan objek yang mengisi kotak dengan sempurna, sementara nilai yang lebih rendah menunjukkan objek yang mungkin memiliki bentuk yang lebih tidak teratur. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Class atau Kelas : Kelas adalah label kategori atau jenis keanggotaan dari objek. Ini adalah informasi klasifikasi yang menunjukkan keberadaan objek dalam kategori tertentu, seperti 0 : kelas 'Rice Cammeo' atau 1 : kelas 'Rice Osmancik'. Jumlah data untuk Cammeo adalah 1630, dan jumlah data untuk kelas Osmancik adalah 2180.")

    st.write("### SUMBER DATASET UCI")
    st.write("Sumber dataset rice cammeo and osmancik.")
    st.write("https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik")

    st.write("### SOURCE CODE APLIKASI DI GOOGLE COLABORATORY")
    st.write("Code dari dataset rice cammeo and osmancik yang diiunputkan ada di google colaboratory di bawah.")
    st.write("https://colab.research.google.com/drive/1CFEcs5tDHGSSE_xyDzEgwABOrnWSy1O5?usp=sharing")

    st.write("### SOURCE CODE APLIKASI DI GITHUB")
    st.write("Code dari dataset rice cammeo and osmancik yang diiunputkan ada di GitHub di bawah.")
    st.write("https://github.com/NabilaAtiraQurratulAini/MLRice.git")

with data_file:
    st.write("### DATA RICE CAMMEO & OSMANCIK")
    st.write("Data")
    df = pd.read_csv("https://raw.githubusercontent.com/NabilaAtiraQurratulAini/PsdA/main/Rice_Osmancik_Cammeo_Dataset.csv")
    df

    column_names = df.columns
    st.write("Nama-nama Kolom dalam Data")
    st.write(column_names)

with preprocessing:
    st.write("### NORMALISASI DATA")
    st.write("Normalisasi adalah proses mengubah nilai-nilai dalam dataset ke dalam skala yang seragam tanpa mengubah struktur relatif di antara nilai-nilai tersebut. Hal ini umumnya dilakukan untuk menghindari perbedaan skala yang besar antara fitur-fitur (kolom-kolom) dalam dataset, yang dapat menyebabkan model machine learning menjadi tidak stabil dan mempengaruhi kinerjanya. Proses normalisasi yang digunakan pada data latih dalam kode ini adalah menggunakan MinMaxScaler. Hasil normalisasi dari data latih disimpan di dalam file 'scaler.pkl' menggunakan modul pickle. Selanjutnya, saat akan melakukan normalisasi data uji, scaler yang telah disimpan tadi di-load kembali menggunakan pickle, dan data uji dinormalisasi menggunakan scaler yang telah di-load. Hasil normalisasi data uji kemudian ditampilkan dalam bentuk DataFrame.")
    st.write("Beberapa metode normalisasi umum termasuk Min-Max Scaling :")
    st.write("### Min-Max Scaling")
    st.write("Xscaled = X - Xmin / Xmax - Xmin")
    st.markdown("""
    Penjelasan :
    - Menyebabkan nilai-nilai dalam dataset berada dalam rentang [0, 1].
    - Cocok untuk data yang memiliki distribusi seragam.
    """)

    df['CLASS'].replace('Cammeo', 0,inplace=True)
    df['CLASS'].replace('Osmancik', 1,inplace=True)

    # langkah 2 : split data menjadi fitur (X) dan target (y)
    X = df.drop(columns=['CLASS'])
    y = df['CLASS']

    # langkah 3 : bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # langkah 4 : normalisasi data menggunakan MinMaxScaler pada data latih
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # langkah 5 : simpan objek scaler ke dalam file "scaler.pkl" menggunakan pickle
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # langkah 6 : load kembali scaler dari file "scaler.pkl" menggunakan pickle
    with open('scaler.pkl', 'rb') as scaler_file:
        scalerr = pickle.load(scaler_file)

    # langkah 7 : normalisasi data uji menggunakan scaler yang telah disimpan
    X_test_scaler = scalerr.transform(X_test)

    # tampilkan hasil normalisasi data uji
    df_test_scaler = pd.DataFrame(X_test_scaler, columns=X.columns)
    st.write(df_test_scaler.head())

with rf:
    st.write("### METODE RANDOM FOREST")
    st.write("Random Forest adalah suatu metode dalam machine learning yang digunakan untuk tugas klasifikasi, regresi, dan masalah-masalah lain yang melibatkan analisis pola atau prediksi. Ini adalah jenis algoritma ensemble, yang berarti itu memadukan hasil dari beberapa model prediksi untuk meningkatkan kinerja dan keakuratannya.")
    
    # membuat dan melatih model Random Forest dengan random sampling
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True)
    random_forest_model.fit(X_train_scaled, y_train)

    # menyimpan model terlatih ke file menggunakan pickle
    with open('random_forest_model.pkl', 'wb') as rf_model_file:
        pickle.dump(random_forest_model, rf_model_file)

    # memuat model terlatih dari file
    with open('random_forest_model.pkl', 'rb') as rf_model_file:
        loaded_rf_model = pickle.load(rf_model_file)

    # membuat prediksi pada data uji
    y_pred_rf = loaded_rf_model.predict(X_test_scaler)

    # evaluasi akurasi model
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    st.write("### HASIL AKURASI")
    st.write("Akurasi Terbaik Menggunakan Metode Random Forest :", accuracy_rf)

    # membuat dataframe untuk membandingkan label sebenarnya dan label yang diprediksi
    rf_results_df = pd.DataFrame({'Actual Label': y_test, 'Prediksi Random Forest': y_pred_rf})
    rf_results_df

    st.write("### CONFUSION MATRIX")

    # hitung confusion matrix
    cm = confusion_matrix(y_test, y_pred_rf)

    # tampilkan confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['CLASS'].unique(), yticklabels=df['CLASS'].unique())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    # menghitung jumlah prediksi yang sesuai dan tidak sesuai
    correct_predictions = rf_results_df['Actual Label'] == rf_results_df['Prediksi Random Forest']
    incorrect_predictions = ~correct_predictions

    # menampilkan jumlah prediksi yang sesuai dan tidak sesuai
    st.write("Jumlah prediksi yang sesuai :", correct_predictions.sum())
    st.write("Jumlah prediksi yang tidak sesuai :", incorrect_predictions.sum())

with inputan:
    st.write("### APLIKASI PREDIKSI JENIS BERAS")

    # formulir input untuk nilai-nilai fitur
    AREA = st.number_input("Masukkan nilai area : ")
    PERIMETER = st.number_input("Masukkan nilai perimeter : ")
    MAJORAXIS = st.number_input("Masukkan nilai majoraxis : ")
    MINORAXIS = st.number_input("Masukkan nilai minoraxis : ")
    ECCENTRICITY = st.number_input("Masukkan nilai eccentricity : ")
    CONVEX_AREA = st.number_input("Masukkan nilai convex area : ")
    EXTENT = st.number_input("Masukkan nilai extent : ")

    # tombol untuk membuat prediksi
    if st.button("Prediksi"):
        # membuat array dengan data input
        new_data = np.array([[AREA, PERIMETER, MAJORAXIS, MINORAXIS, ECCENTRICITY, CONVEX_AREA, EXTENT]])

        # normalisasi data input menggunakan scaler yang disimpan
        new_data_scaled = scalerr.transform(new_data)

        # melakukan prediksi menggunakan model SVM
        prediction = loaded_rf_model.predict(new_data_scaled)

        # menampilkan hasil prediksi
        if prediction[0] == 1:
            st.write("Hasil Prediksi : Beras Osmancik")
        else:
            st.write("Hasil Prediksi : Beras Cammeo")
