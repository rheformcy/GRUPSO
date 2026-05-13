# =========================================================
# IMPORT LIBRARY & REPRODUCIBILITY SETTINGS
# =========================================================
import os

# Wajib di paling atas sebelum import TF
os.environ["PYTHONHASHSEED"] = "49"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import gc

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from pyswarms.single.global_best import GlobalBestPSO

# SEED GLOBAL
SEED = 49

def reset_seeds():
    """Fungsi untuk meriset semua seed ke kondisi awal agar hasil deterministik"""
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

reset_seeds()

# =========================================================
# STREAMLIT UI CONFIG
# =========================================================
st.set_page_config(
    page_title="GRU-PSO Precision Gold Forecasting",
    layout="wide"
)

st.title("📈 Forecasting Harga Emas: GRU-PSO (Precision Sync)")
st.markdown("Versi ini dioptimasi agar hasilnya **identik** dengan Google Colab.")

# Tampilkan informasi versi untuk cross-check
st.sidebar.info(f"TF Version: {tf.__version__} | NP Version: {np.__version__}")

# =========================================================
# SIDEBAR PARAMETERS
# =========================================================
st.sidebar.header("📂 Data & Parameter")
uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx", "xls"])

timestep = st.sidebar.number_input("Timestep", min_value=1, value=1)
particles = st.sidebar.number_input("Jumlah Partikel PSO", min_value=1, value=40)
iterasi = st.sidebar.number_input("Jumlah Iterasi PSO", min_value=1, value=10)
epochs_pso = st.sidebar.number_input("Epoch saat PSO", min_value=1, value=10)
epochs_final = st.sidebar.number_input("Epoch Final Training", min_value=1, value=50)

st.sidebar.divider()
st.sidebar.header("🎛️ Hyperparameter Range")
units_min = st.sidebar.number_input("Units Min", value=16)
units_max = st.sidebar.number_input("Units Max", value=128)
lr_min = st.sidebar.number_input("LR Min", format="%.4f", value=0.0001)
lr_max = st.sidebar.number_input("LR Max", format="%.4f", value=0.01)
batch_min = st.sidebar.number_input("Batch Min", value=16)
batch_max = st.sidebar.number_input("Batch Max", value=128)
dropout_min = st.sidebar.slider("Dropout Min", 0.0, 0.9, 0.1)
dropout_max = st.sidebar.slider("Dropout Max", 0.0, 0.9, 0.5)

start_button = st.sidebar.button("🚀 Jalankan Optimasi", use_container_width=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# =========================================================
# MAIN PROCESSING
# =========================================================
if uploaded_file is not None:
    # Load Data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Cleaning Numeric (Menangani ribuan yang mungkin terbaca string)
    if df["Terakhir"].dtype == 'O':
        df["Terakhir"] = df["Terakhir"].str.replace('.', '').str.replace(',', '.')
    df["Terakhir"] = pd.to_numeric(df["Terakhir"], errors="coerce")
    df = df.dropna(subset=["Terakhir"]).reset_index(drop=True)

    st.subheader("📄 Preview Data Bersih")
    st.dataframe(df.head(), use_container_width=True)

    if start_button:
        # 1. Preprocessing & Scaling
        # Penting: Fit scaler HANYA pada data train (80%)
        data_raw = df[["Terakhir"]].values
        split_idx = int(len(data_raw) * 0.8)
        
        scaler = MinMaxScaler().fit(data_raw[:split_idx])
        data_scaled = scaler.transform(data_raw)
        
        # 2. Windowing
        X_all, y_all = make_sequences(data_scaled, timestep)
        
        # Penyesuaian index setelah windowing
        train_end_idx = split_idx - timestep
        X_train_full = X_all[:train_end_idx]
        y_train_full = y_all[:train_end_idx]
        X_test = X_all[train_end_idx:]
        y_test = y_all[train_end_idx:]
        
        # 3. Manual Split Validation (Agar Identik)
        # Jangan pakai validation_split di fit(), tapi split manual di sini
        val_size = int(len(X_train_full) * 0.2)
        tr_size = len(X_train_full) - val_size
        
        X_tr = X_train_full[:tr_size]
        y_tr = y_train_full[:tr_size]
        X_val = X_train_full[tr_size:]
        y_val = y_train_full[tr_size:]

        # =================================================
        # FITNESS FUNCTION PSO
        # =================================================
        def objective_function(particles_array):
            n_particles = particles_array.shape[0]
            losses = np.zeros(n_particles)
            
            for i, p in enumerate(particles_array):
                u = int(np.round(p[0]))
                lr = p[1]
                b = int(np.round(p[2]))
                d = p[3]
                
                # Reset environment setiap partikel
                clear_session()
                reset_seeds()
                
                model = Sequential([
                    Input(shape=(X_tr.shape[1], 1)),
                    GRU(u, activation="tanh", kernel_initializer='glorot_uniform'),
                    Dropout(d),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
                
                model.fit(
                    X_tr, y_tr, 
                    epochs=epochs_pso, 
                    batch_size=max(1, b), 
                    verbose=0, 
                    shuffle=False
                )
                
                pred = model.predict(X_val, verbose=0)
                # Evaluasi MSE pada skala asli untuk fitness
                p_inv = scaler.inverse_transform(pred)
                a_inv = scaler.inverse_transform(y_val)
                losses[i] = mean_squared_error(a_inv, p_inv)
                
                gc.collect()
            return losses

        # =================================================
        # RUN PSO
        # =================================================
        status = st.empty()
        status.info("🚀 Sedang mencari hyperparameter terbaik...")
        
        bounds = (
            np.array([units_min, lr_min, batch_min, dropout_min]),
            np.array([units_max, lr_max, batch_max, dropout_max])
        )
        
        optimizer = GlobalBestPSO(
            n_particles=particles, 
            dimensions=4, 
            options={'c1': 2.0, 'c2': 2.0, 'w': 0.7}, 
            bounds=bounds
        )
        
        best_cost, best_pos = optimizer.optimize(objective_function, iters=iterasi)
        
        # Best Params
        b_u, b_lr, b_b, b_d = int(np.round(best_pos[0])), best_pos[1], int(np.round(best_pos[2])), best_pos[3]

        # =================================================
        # FINAL TRAINING
        # =================================================
        status.info("🤖 Melakukan Training Final dengan Parameter Terbaik...")
        clear_session()
        reset_seeds()

        final_model = Sequential([
            Input(shape=(X_train_full.shape[1], 1)),
            GRU(b_u, activation="tanh", kernel_initializer='glorot_uniform'),
            Dropout(b_d),
            Dense(1)
        ])
        final_model.compile(optimizer=Adam(learning_rate=b_lr), loss="mse")

        history = final_model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val), # Gunakan data validasi manual
            epochs=epochs_final,
            batch_size=b_b,
            shuffle=False,
            verbose=1
        )

        # =================================================
        # EVALUATION & RESULTS
        # =================================================
        status.success("✅ Selesai!")
        
        y_pred_scaled = final_model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_actual = scaler.inverse_transform(y_test).flatten()

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        mape = mean_absolute_percentage_error(y_actual, y_pred) * 100

        # Display
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Units", b_u)
        col2.metric("Best LR", f"{b_lr:.4f}")
        col3.metric("Best Batch", b_b)
        col4.metric("Best Dropout", f"{b_d:.2f}")

        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE (Rp)", f"{rmse:,.2f}")
        m2.metric("MAE (Rp)", f"{mae:,.2f}")
        m3.metric("MAPE", f"{mape:.4f}%")

        # Visualization
        st.subheader("📈 Grafik Hasil Prediksi (Data Test)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_actual, label="Harga Aktual", color="blue", linewidth=2)
        ax.plot(y_pred, label="Harga Prediksi", color="red", linestyle="--", linewidth=2)
        ax.set_title("Perbandingan Aktual vs Prediksi")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

else:
    st.info("👋 Halo Rhena! Silakan upload dataset emas kamu untuk memulai.")
