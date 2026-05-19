import streamlit as st
import tensorflow as tf
import random
import os

SEED = 49

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Input, GRU, Dropout, Dense
from keras.optimizers import Adam
from keras.backend import clear_session
import gc
from pyswarms.single import GlobalBestPSO

# ==========================================
# 1. KUNCI ALL SEEDS DI AWAL SKRIP (SAMA PERSIS)
# ==========================================
def reset_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tf.config.experimental.enable_op_determinism()

st.title("Aplikasi Prediksi Harga Emas GRU-ADAM")

# Input File dari User
uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Membaca data dengan aman
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    st.success("Data berhasil diunggah!")
    
    # ==========================================
    # 2. PROSES PEMODELAN DIKUNCI DI DALAM CACHE
    # ==========================================
    # Diubah ke cache_data agar aman menyimpan output list numerik murni tanpa crash graph Keras
    @st.cache_data
    def jalankan_gru_standar_pure(_df_emas):
        # Reset seed tepat sebelum pemrosesan data dimulai
        clear_session()
        reset_seeds()
        
        # Penyiapan fitur
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        # Split Data Training dan Testing (80:20)
        values = _df_emas[['Terakhir']].values
        n = len(values)
        n_train = int(n * 0.8)

        # Data Scaling
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        # Fungsi Windowing Asli Kamu
        def make_sequences(X_scaled, y_scaled, window):
            X_seq, y_seq = [], []
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
                y_seq.append(y_scaled[i])
            return np.array(X_seq), np.array(y_seq)
    
        # Windowing Data
        window = 1
        X_seq_all, y_seq_all = make_sequences(Xs, ys, window)
        
        # Split train-test
        dtrain_end = n_train - window
        
        X_train = X_seq_all[:dtrain_end]
        y_train = y_seq_all[:dtrain_end]
        
        X_test = X_seq_all[dtrain_end:]
        y_test = y_seq_all[dtrain_end:]
        
        # Reshape untuk input GRU
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # --- PARAMETER ARSITEKTUR ADAM STANDAR (DARI COLAB) ---
        GS_epoch = 50
        GS_batch = 32
        GS_units = 16
        GS_dropout = 0.0
        GS_LR = 0.001
        
        # --- BANGUN STRUKTUR MODEL DARI NOL ---
        reset_seeds() 
        model_std = Sequential()
        model_std.add(Input(shape=(window, 1)))
        model_std.add(GRU(units=GS_units, activation='tanh'))
        model_std.add(Dropout(GS_dropout))
        model_std.add(Dense(units=1, activation='linear'))
        model_std.compile(optimizer=Adam(learning_rate=GS_LR), loss='mse')
        
        # KUNCI UTAMA: shuffle=False dikembalikan agar urutan waktu konsisten seperti di Colab
        model_std.fit(
            X_train, y_train,
            epochs=GS_epoch,
            batch_size=GS_batch,
            validation_split=0.2,
            verbose=0
        )

        # --- PROSES PREDIKSI DATA TESTING ---
        y_pred_scaled = model_std.predict(X_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # KUNCI KEDUA: Ambil data aktual via inverse transform sekuensial agar sejajar & anti-geser tanggal
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Selaraskan ukuran array jika ada selisih efek pembulatan windowing
        min_len = min(len(y_test_inv), len(y_pred_inv))
        y_test_inv = y_test_inv[:min_len]
        y_pred_inv = y_pred_inv[:min_len]
        
        # Hitung Nilai Metrik Evaluasi Akhir
        std_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        std_mae = mean_absolute_error(y_test_inv, y_pred_inv)
        std_mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        # Return dalam bentuk list numerik agar lolos standarisasi serialisasi cache Streamlit
        return GS_units, GS_LR, GS_batch, GS_dropout, std_rmse, std_mae, std_mape, y_test_inv.tolist(), y_pred_inv.tolist()

    # ----------------------------------------------------
    # TOMBOL INTERFACE WEB STREAMLIT
    # ----------------------------------------------------
    if st.button("Mulai Pemrosesan Model Adam Standar"):
        with st.spinner("Sedang melatih model GRU full 50 epoch tanpa henti... Mohon tunggu."):
            std_units, std_lr, std_batch, std_dropout, rmse_s, mae_s, mape_s, y_true_list, y_pred_list = jalankan_gru_standar_pure(emas)
        st.success("Proses Training Selesai!")
        
        # Kembalikan tipe list dari cache ke numpy array untuk kebutuhan visualisasi plot grafik
        y_true_s = np.array(y_true_list)
        y_pred_s = np.array(y_pred_list)
        
        # Tampilkan Parameter Adam Standar
        st.subheader("Arsitektur & Hyperparameter Model (Adam Standar):")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units GRU", std_units)
        col2.metric("Learning Rate", f"{std_lr:.3f}")
        col3.metric("Batch Size", std_batch)
        col4.metric("Dropout", f"{std_dropout:.1f}")
        
        # Tampilkan Hasil Evaluasi Metrik
        st.subheader("Hasil Evaluasi Data Testing (Adam Standar):")
        res_df = pd.DataFrame([{
            'RMSE (Rp)': round(rmse_s, 2),
            'MAE (Rp)': round(mae_s, 2),
            'MAPE (%)': round(mape_s, 4)
        }])
        st.dataframe(res_df, use_container_width=True)

        # Plot Hasil Prediksi Adam Standar
        st.subheader("Visualisasi Grafik Prediksi (Adam Standar)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true_s, label='Harga Aktual', color='royalblue', linewidth=2)
        ax.plot(y_pred_s, label='Harga Prediksi Adam', color='darkorange', linestyle='--', linewidth=2)
        ax.set_title("Perbandingan Harga Aktual vs Prediksi (GRU Adam Standar)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
