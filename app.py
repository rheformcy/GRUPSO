import streamlit as st
import tensorflow as tf
import random
import os

# Set Seed Global tingkat atas agar deterministik sesuai Colab
SEED = 49

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Input, GRU, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.backend import clear_session

def reset_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.experimental.enable_op_determinism()

st.title("Aplikasi Prediksi Harga Emas GRU-ADAM (Pure Colab Simulator)")

# Input File dari User
uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    st.success("Data berhasil diunggah!")

    # ==========================================
    # PROSES PEMODELAN PURE SEPERTI DI COLAB
    # ==========================================
    @st.cache_data
    def jalankan_pemodelan_colab_pure(_df_emas):
        clear_session()
        reset_seeds()
        
        # Penyiapan fitur berdasarkan kolom tunggal "Terakhir"
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        # Split Data Training dan Testing (80:20)
        values = _df_emas[['Terakhir']].values
        n = len(values)
        n_train = int(n * 0.8)

        # Data Scaling (Fit hanya pada data training saja)
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        # Fungsi Windowing Sekuensial
        def make_sequences(X_scaled, y_scaled, window):
            X_seq, y_seq = [], []
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
                y_seq.append(y_scaled[i])
            return np.array(X_seq), np.array(y_seq)
    
        GS_window = 1
        X_seq_all, y_seq_all = make_sequences(Xs, ys, GS_window)
        
        # Split data sequence sesuai indeks Colab
        dtrain_end = n_train - GS_window
        X_train = X_seq_all[:dtrain_end]
        y_train = y_seq_all[:dtrain_end]
        X_test  = X_seq_all[dtrain_end:]
        y_test  = y_seq_all[dtrain_end:]
        
        # Reshape murni input 3D [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Inisialisasi Parameter Model (Saklek dari Skrip Colab Kamu)
        GS_epoch = 50
        GS_batch = 32
        GS_units = 16
        GS_layers = 1
        GS_dropout = 0.0
        GS_LR = 0.001
        
        # Fungsi pembangun model dengan paksaan parameter reset_after=True sesuai isi file .h5 kamu
        def build_gru_model(units, layers, dropout, lr, window):
            n_features = 1
            model = Sequential()
            model.add(Input(shape=(window, n_features)))
            if layers == 1:
                # KUNCI UTAMA: Ditambahkan reset_after=True secara eksplisit
                model.add(GRU(units=units, activation='tanh', recurrent_activation='sigmoid', reset_after=True))
                model.add(Dropout(dropout))
            else:
                for i in range(layers):
                    is_last = (i == layers - 1)
                    model.add(GRU(units=units, return_sequences=not is_last, activation='tanh', recurrent_activation='sigmoid', reset_after=True))
                    model.add(Dropout(dropout))
            model.add(Dense(units=1, activation='linear'))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            return model

        # Bangun & latih model murni dari nol
        reset_seeds()
        gru_standar = build_gru_model(GS_units, GS_layers, GS_dropout, GS_LR, GS_window)
      
        history = gru_standar.fit(
            X_train, y_train,
            epochs=GS_epoch,
            batch_size=GS_batch,
            validation_split=0.2,
            verbose=0  
        )

        epoch_stopped = len(history.history['loss'])
        
        # Metriks Evaluasi Akhir (Sama Persis Rumus Colab)
        y_pred_scaled = gru_standar.predict(X_test, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae  = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        return GS_units, GS_LR, GS_batch, rmse, mae, mape, epoch_stopped, y_test_inv.tolist(), y_pred_inv.tolist()

    # ----------------------------------------------------
    # TOMBOL EKSEKUSI INTERFACE
    # ----------------------------------------------------
    if st.button("Mulai Pemrosesan Model Adam Standar"):
        with st.spinner("Menjalankan Pure Training Simulasi Sinkronisasi Manifes..."):
            units, lr, batch, rmse_s, mae_s, mape_s, ep_stopped, y_true_list, y_pred_list = jalankan_pemodelan_colab_pure(emas)
            
        st.success("Proses Training Selesai!")
        
        y_true_s = np.array(y_true_list)
        y_pred_s = np.array(y_pred_list)
        
        # Tampilkan Parameter Adam Standar
        st.subheader("Arsitektur & Hyperparameter Model:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units GRU", units)
        col2.metric("Learning Rate", f"{lr:.3f}")
        col3.metric("Batch Size", batch)
        col4.metric("Epoch Berhenti (Early Stop)", ep_stopped)
        
        # Tampilkan Hasil Evaluasi Metrik
        st.subheader("Hasil Evaluasi Data Testing:")
        res_df = pd.DataFrame([{
            'RMSE (Rp)': round(rmse_s, 2),
            'MAE (Rp)': round(mae_s, 2),
            'MAPE (%)': round(mape_s, 4)
        }])
        st.dataframe(res_df, use_container_width=True)

        # Plot Hasil Prediksi Adam Standar
        st.subheader("Visualisasi Grafik Prediksi")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true_s, label='Harga Aktual', color='royalblue', linewidth=2)
        ax.plot(y_pred_s, label='Harga Prediksi Adam', color='darkorange', linestyle='--', linewidth=2)
        ax.set_title("Perbandingan Harga Aktual vs Prediksi (Pure Colab Verified)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
