import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Prediksi Harga Emas GRU Standar", layout="wide")

st.title("Aplikasi Prediksi Harga Emas GRU Standar")

# Input File dari User
uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    st.success("Data berhasil diunggah!")
    
    st.subheader("Konfigurasi Model")
    st.info("Aplikasi berjalan dalam mode Sinkronisasi Bobot (Membuat Scaler Lokal & Menyuntikkan Bobot .h5)")

    # =========================================================
    # FUNGSI UTAMA MODEL GRU STANDAR (TANPA JOBLIB)
    # =========================================================
    @st.cache_resource
    def jalankan_gru_standar(_df_emas):
        from keras.models import Sequential, load_model
        from keras.layers import Input, GRU, Dropout, Dense
        from keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        
        # --- 1. PRAPEMROSESAN DATA (LOGIKA KEMBAR IDENTIK DENGAN COLAB) ---
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        values = _df_emas[['Terakhir']].values
        n = len(values)
        n_train = int(n * 0.8)
        
        # Membuat dan mengepas (fit) scaler lokal menggunakan porsi data training asli
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        # Membuat susunan sequence windowing (Timestep = 1)
        window = 1
        X_seq, y_seq = [], []
        for i in range(window, len(ys)):
            X_seq.append(Xs[i-window:i])
            y_seq.append(ys[i])
        
        X_seq_all, y_seq_all = np.array(X_seq), np.array(y_seq)
        dtrain_end = n_train - window
        
        # Potong porsi data testing secara eksak
        X_test = X_seq_all[dtrain_end:]
        y_test = y_seq_all[dtrain_end:]
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # --- 2. AMBIL ARSITEKTUR LOKAL & SUNTIK BOBOT .H5 COLAB ---
        nama_file_model = 'Best Model STD (TW) Timestep -- 1.h5'
        if os.path.exists(nama_file_model):
            # Membangun struktur kosong standar di internal Streamlit
            gru_standar = Sequential()
            gru_standar.add(Input(shape=(window, 1)))
            gru_standar.add(GRU(units=16, activation='tanh'))
            gru_standar.add(Dropout(0.0))
            gru_standar.add(Dense(units=1, activation='linear'))
            gru_standar.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Ekstrak matriks bobot numerik dari berkas h5 tanpa menghiraukan konfigurasi versi layer Keras
            try:
                model_colab = load_model(nama_file_model, compile=False)
                gru_standar.set_weights(model_colab.get_weights())
            except Exception as load_err:
                gru_standar.load_weights(nama_file_model, by_name=True, skip_mismatch=True)
        else:
            st.error(f"File model '{nama_file_model}' tidak ditemukan di folder aplikasi! Harap unduh berkas .h5 dari Google Drive skripsimu.")
            st.stop()

        # --- 3. PROSES PREDIKSI DATA TESTING ---
        y_pred_scaled = gru_standar.predict(X_test, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Hitung Nilai Metrik Evaluasi Akhir
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        return 16, 0.001, 32, 0.0, rmse, mae, mape, y_test_inv, y_pred_inv

    # ----------------------------------------------------
    # TOMBOL EKSEKUSI PADA INTERFACE WEB
    # ----------------------------------------------------
    if st.button("Mulai Pemrosesan Model Adam"):
        with st.spinner("Sedang memuat model dan memproses data... Mohon tunggu."):
            units, lr, batch, dropout, rmse, mae, mape, y_true_plot, y_pred_plot = jalankan_gru_standar(emas)
        st.success("Proses Evaluasi GRU Standar Selesai!")
        
        # 1. Tampilkan Arsitektur Model ke Interface Web
        st.subheader("Arsitektur & Hyperparameter Model (Sesuai Buku Skripsi):")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units GRU", units)
        col2.metric("Learning Rate", f"{lr:.3f}")
        col3.metric("Batch Size", batch)
        col4.metric("Dropout", f"{dropout:.1f}")
        
        # 2. Tampilkan Hasil Evaluasi Metrik
        st.subheader("Hasil Evaluasi Data Testing:")
        res_df = pd.DataFrame([{
            'RMSE (Rp)': round(rmse, 2),
            'MAE (Rp)': round(mae, 2),
            'MAPE (%)': round(mape, 4)
        }])
        st.dataframe(res_df, use_container_width=True)

        # 3. Plot Hasil Perbandingan ke Interface Web
        st.subheader("Visualisasi Grafik Prediksi")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true_plot, label='Harga Aktual', color='royalblue', linewidth=2)
        ax.plot(y_pred_plot, label='Harga Prediksi', color='crimson', linestyle='--', linewidth=2)
        ax.set_title("Perbandingan Harga Aktual vs Prediksi (GRU Adam Standar)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
