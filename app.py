import streamlit as st
import os

# --- LANGKAH CRITICAL 1: SET ENVIRONMENT VARIABLES SEBELUM TENSORFLOW DI-IMPORT ---
# Ini wajib ditaruh di paling atas skrip agar TensorFlow patuh sejak awal lahir.
SEED = 49
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Memaksa Streamlit memakai CPU standar agar sama dengan CPU hitungan lokal

import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- LANGKAH CRITICAL 2: KUNCI ALL SEEDS DI LEVEL GLOBAL ---
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
if hasattr(tf.config.experimental, 'enable_op_determinism'):
    tf.config.experimental.enable_op_determinism()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Input, GRU, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.backend import clear_session

# Fungsi pembantu untuk reset ulang seed saat rerun fungsi Streamlit
def reset_seeds_local():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

st.title("Aplikasi Prediksi Harga Emas GRU Standar")

uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    st.success("Data berhasil diunggah!")
    st.subheader("Konfigurasi Model")
    st.info("Model berjalan menggunakan Optimizer Adam Standar (Baseline Model)")

    # ==========================================
    # 2. FUNGSI UTAMA TRAINING GRU STANDAR
    # ==========================================
    @st.cache_resource
    def jalankan_gru_standar(_df_emas):
        # Bersihkan session memori sisa rerun sebelumnya dan tegaskan seed kembali
        clear_session()
        reset_seeds_local()
        
        # --- PRAPEMROSESAN DATA (LOGIKA KEMBARAN COLAB) ---
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        values = _df_emas[['Terakhir']].values
        n = len(values)
        n_train = int(n * 0.8)
        
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        window = 1
        def make_sequences(X_scaled, y_scaled, window=1):
            X_seq, y_seq = [], []
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
                y_seq.append(y_scaled[i])
            return np.array(X_seq), np.array(y_seq)
    
        X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)
        dtrain_end = n_train - window
        
        X_train = X_seq_all[:dtrain_end]
        y_train = y_seq_all[:dtrain_end]
        X_test = X_seq_all[dtrain_end:]
        y_test = y_seq_all[dtrain_end:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # --- INITIALIZATION PARAMETER ---
        GS_epoch = 50
        GS_batch = 32
        GS_units = 16
        GS_layers = 1
        GS_dropout = 0.0
        GS_LR = 0.001
        GS_window = 1
        
        # --- FUNGSI BUILD MODEL (STRUKTUR REVISI BERSIH) ---
        def build_gru_model(units, layers, dropout, lr, window):
            n_features = 1
            model = Sequential()
            model.add(Input(shape=(window, n_features)))
            if layers == 1:
                model.add(GRU(units=units, activation='tanh'))
                model.add(Dropout(dropout))
            else:
                for i in range(layers):
                    is_last = (i == layers - 1)
                    model.add(GRU(units=units, return_sequences=not is_last, activation='tanh'))
                    model.add(Dropout(dropout))
            model.add(Dense(units=1, activation='linear'))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            return model
        
        # Eksekusi pembuatan model secara linear setelah seed ditegaskan ulang
        reset_seeds_local()
        gru_standar = build_gru_model(GS_units, GS_layers, GS_dropout, GS_LR, GS_window)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        # Eksekusi Fit dengan mematikan pengacakan (shuffle=False)
        gru_standar.fit(
            X_train, y_train,
            epochs=GS_epoch,
            batch_size=GS_batch,
            callbacks=[early_stop],
            validation_split=0.2,
            verbose=1,
            shuffle=False 
        )

        # --- EVALUASI METRIK ---
        y_pred_scaled = gru_standar.predict(X_test, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        return GS_units, GS_LR, GS_batch, GS_dropout, rmse, mae, mape, y_test_inv, y_pred_inv

    # ----------------------------------------------------
    # TOMBOL INTERFACE
    # ----------------------------------------------------
    if st.button("Mulai Pemrosesan Model Adam"):
        with st.spinner("Sedang melatih model GRU Standar dengan Adam... Mohon tunggu."):
            units, lr, batch, dropout, rmse, mae, mape, y_true_plot, y_pred_plot = jalankan_gru_standar(emas)
        st.success("Eksekusi GRU Standar Selesai!")
        
        st.subheader("Arsitektur & Hyperparameter Model:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units GRU", units)
        col2.metric("Learning Rate", f"{lr:.6f}")
        col3.metric("Batch Size", batch)
        col4.metric("Dropout", f"{dropout:.4f}")
        
        st.subheader("Hasil Evaluasi Data Testing:")
        res_df = pd.DataFrame([{
            'RMSE (Rp)': round(rmse, 2),
            'MAE (Rp)': round(mae, 2),
            'MAPE (%)': round(mape, 4)
        }])
        st.dataframe(res_df)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true_plot, label='Harga Aktual', color='royalblue', linewidth=2)
        ax.plot(y_pred_plot, label='Harga Prediksi', color='crimson', linestyle='--', linewidth=2)
        ax.set_title("Perbandingan Harga Aktual vs Prediksi (GRU Adam Standar)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
