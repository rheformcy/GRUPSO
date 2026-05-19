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
from keras.callbacks import EarlyStopping
from keras.backend import clear_session
import gc
from pyswarms.single import GlobalBestPSO

# ==========================================
# 1. KUNCI ALL SEEDS DI AWAL SKRIP
# ==========================================
def reset_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.experimental.enable_op_determinism()

st.title("Aplikasi Perbandingan Prediksi Harga Emas GRU")

# Input File dari User
uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    st.success("Data berhasil diunggah!")
    
    # ----------------------------------------------------
    # PILIHAN ALGORITMA OPTIMASI (MENU DROPDOWN)
    # ----------------------------------------------------
    st.subheader("Konfigurasi Model")
    pilihan_optimasi = st.selectbox(
        "Pilih Metode Optimasi Hyperparameter:",
        ("Optimizer Adam Standar (Baseline Model)", "Optimasi Metaheuristik PSO (Proposed Model)")
    )

    # ==========================================
    # 2. FUNGSI MODEL GRU STANDAR (ADAM BATCH=32, UNITS=16)
    # ==========================================
    @st.cache_resource
    def jalankan_gru_standar(_df_emas):
        reset_seeds()
        
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        n = len(data_features)
        n_train = int(n * 0.8)
        
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        def make_sequences(X_scaled, y_scaled, window=1):
            X_seq, y_seq = [], []
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
                y_seq.append(y_scaled[i])
            return np.array(X_seq), np.array(y_seq)
    
        X_seq_all, y_seq_all = make_sequences(Xs, ys, window=1)
        dtrain_end = n_train - 1
        X_train = X_seq_all[:dtrain_end]
        y_train = y_seq_all[:dtrain_end]
        X_test = X_seq_all[dtrain_end:]
        y_test = y_seq_all[dtrain_end:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Parameter sesuai Google Colab Standar kamu
        gru_standar = Sequential([
            Input(shape=(1, 1)),
            GRU(units=16, activation='tanh'),
            Dropout(0.0),
            Dense(1, activation='linear')
        ])
        gru_standar.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Memakai EarlyStopping bawaan kode Colab kamu
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        history = gru_standar.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            validation_split=0.2,
            verbose=1
        )
        
        y_pred_scaled = gru_standar.predict(X_test, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        return 16, 0.001, 32, 0.0, rmse, mae, mape, y_test_inv, y_pred_inv

    # ==========================================
    # 3. FUNGSI MODEL GRU-PSO (SAMA SEPERTI SEBELUMNYA)
    # ==========================================
    @st.cache_resource
    def jalankan_pemodelan_pso_gru(_df_emas):
        reset_seeds()
        
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        n = len(data_features)
        n_train = int(n * 0.8)
        
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        def make_sequences(X_scaled, y_scaled, window=1):
            X_seq, y_seq = [], []
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
                y_seq.append(y_scaled[i])
            return np.array(X_seq), np.array(y_seq)
    
        X_seq_all, y_seq_all = make_sequences(Xs, ys, window=1)
        dtrain_end = n_train - 1
        X_train = X_seq_all[:dtrain_end]
        y_train = y_seq_all[:dtrain_end]
        X_test = X_seq_all[dtrain_end:]
        y_test = y_seq_all[dtrain_end:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        val_PSOSL = 0.2
        n_tr_samples_PSOSL = X_train.shape[0]
        n_tr_val_PSOSL = int(n_tr_samples_PSOSL * (1 - val_PSOSL))
        
        X_tr_PSOSL = X_train[:n_tr_val_PSOSL]
        y_tr_PSOSL = y_train[:n_tr_val_PSOSL]
        X_val_PSOSL = X_train[n_tr_val_PSOSL:]
        y_val_PSOSL = y_train[n_tr_val_PSOSL:]

        def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y):
            def obj_fn(particles):
                n_particles = particles.shape[0]
                costs = np.zeros(n_particles)
                for i, p in enumerate(particles):
                    units = int(np.round(p[0]))
                    lr = float(p[1])
                    batch = int(np.round(p[2]))
                    dropout = float(p[3])
                    try:
                        tf.random.set_seed(49)
                        clear_session()
                        model = Sequential([
                            Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                            GRU(units=units, activation='tanh'),
                            Dropout(dropout),
                            Dense(1)
                        ])
                        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                        model.fit(X_tr, y_tr, epochs=10, batch_size=batch, verbose=0)
            
                        yv_pred = model.predict(X_va, verbose=0)
                        yv_pred_orig_PSOSL = scaler_y.inverse_transform(yv_pred).flatten()
                        yv_true_orig_PSOSL = scaler_y.inverse_transform(y_va.reshape(-1, 1)).flatten()
                        costs[i] = mean_squared_error(yv_true_orig_PSOSL, yv_pred_orig_PSOSL)
                    except Exception:
                        costs[i] = 1e12
                    clear_session()
                    gc.collect()
                return costs
            return obj_fn
            
        pso_obj_PSOSL = make_pso_obj(X_tr_PSOSL, y_tr_PSOSL, X_val_PSOSL, y_val_PSOSL, scaler_y)

        optimizer = GlobalBestPSO(
            n_particles=40, dimensions=4,
            options={'c1': 2.0, 'c2': 2.0, 'w': 0.7},
            bounds=([16, 0.0001, 16, 0.01], [128, 0.01, 128, 0.5])
        )
        
        n_particles, dims = optimizer.swarm.position.shape
        optimizer.swarm.pbest_pos_PSOSL = optimizer.swarm.position.copy()
        optimizer.swarm.pbest_cost_PSOSL = np.full(n_particles, np.inf)
        
        history_gbest_cost_PSOSL = []
        history_gbest_pos_PSOSL = []
        
        for it in range(1):
            costs_PSOSL = pso_obj_PSOSL(optimizer.swarm.position)
            mask_PSOSL = costs_PSOSL < optimizer.swarm.pbest_cost_PSOSL
            optimizer.swarm.pbest_cost_PSOSL[mask_PSOSL] = costs_PSOSL[mask_PSOSL]
            optimizer.swarm.pbest_pos_PSOSL[mask_PSOSL] = optimizer.swarm.position[mask_PSOSL].copy()
            
            best_PSOSL = np.argmin(optimizer.swarm.pbest_cost_PSOSL)
            optimizer.swarm.best_cost_PSOSL = optimizer.swarm.pbest_cost_PSOSL[best_PSOSL]
            optimizer.swarm.best_pos_PSOSL = optimizer.swarm.pbest_pos_PSOSL[best_PSOSL].copy()
            
            history_gbest_cost_PSOSL.append(float(optimizer.swarm.best_cost_PSOSL))
            history_gbest_pos_PSOSL.append(optimizer.swarm.best_pos_PSOSL.copy())
            
            r1 = np.random.rand(*optimizer.swarm.position.shape)
            r2 = np.random.rand(*optimizer.swarm.position.shape)
            optimizer.swarm.velocity = (
                0.7 * optimizer.swarm.velocity
                + 2.0 * r1 * (optimizer.swarm.pbest_pos_PSOSL - optimizer.swarm.position)
                + 2.0 * r2 * (optimizer.swarm.best_pos_PSOSL - optimizer.swarm.position)
            )
            optimizer.swarm.position += optimizer.swarm.velocity
            optimizer.swarm.position = np.clip(optimizer.swarm.position, np.array([16, 0.0001, 16, 0.01]), np.array([128, 0.01, 128, 0.5]))

        best_pos_PSOSL = history_gbest_pos_PSOSL[-1]
        best_units_PSOSL = int(np.round(best_pos_PSOSL[0]))
        best_lr_PSOSL = float(best_pos_PSOSL[1])
        best_batch_PSOSL = int(np.round(best_pos_PSOSL[2]))
        best_dropout_PSOSL = float(best_pos_PSOSL[3])

        tf.random.set_seed(49)
        GRU_PSOSL = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            GRU(units=best_units_PSOSL, activation='tanh'),
            Dropout(best_dropout_PSOSL),
            Dense(1)
        ])
        GRU_PSOSL.compile(optimizer=Adam(learning_rate=best_lr_PSOSL), loss='mse')
        
        history_final = GRU_PSOSL.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=best_batch_PSOSL, 
            validation_split=0.2, 
            verbose=1
        )
        
        y_pred_PSOSL = GRU_PSOSL.predict(X_test, verbose=0)
        y_pred_orig_PSOSL = scaler_y.inverse_transform(y_pred_PSOSL).flatten()
        y_test_orig_PSOSL = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse_PSOSL = np.sqrt(mean_squared_error(y_test_orig_PSOSL, y_pred_orig_PSOSL))
        mae_PSOSL = mean_absolute_error(y_test_orig_PSOSL, y_pred_orig_PSOSL)
        mape_PSOSL = mean_absolute_percentage_error(y_test_orig_PSOSL, y_pred_orig_PSOSL) * 100

        return (
            best_units_PSOSL,
            best_lr_PSOSL,
            best_batch_PSOSL,
            best_dropout_PSOSL,
            rmse_PSOSL,
            mae_PSOSL,
            mape_PSOSL,
            y_test_orig_PSOSL,
            y_pred_orig_PSOSL
        )

    # ----------------------------------------------------
    # LOGIKA TOMBOL EKSEKUSI BERDASARKAN PILIHAN DROPDOWN
    # ----------------------------------------------------
    if st.button("Mulai Pemrosesan Model"):
        if pilihan_optimasi == "Optimizer Adam Standar (Baseline Model)":
            with st.spinner("Sedang melatih model GRU Standar dengan Adam... Mohon tunggu."):
                units, lr, batch, dropout, rmse, mae, mape, y_true_plot, y_pred_plot = jalankan_gru_standar(emas)
            st.success("Eksekusi GRU Standar Selesai!")
            
        else:
            with st.spinner("Sedang mencari hyperparameter terbaik menggunakan PSO... Mohon tunggu."):
                units, lr, batch, dropout, rmse, mae, mape, y_true_plot, y_pred_plot = jalankan_pemodelan_pso_gru(emas)
            st.success("Eksekusi GRU-PSO Selesai!")
        
        # Tampilkan Parameter ke Interface Web
        st.subheader("Arsitektur & Hyperparameter Model yang Digunakan:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units GRU", units)
        col2.metric("Learning Rate", f"{lr:.6f}")
        col3.metric("Batch Size", batch)
        col4.metric("Dropout", f"{dropout:.4f}")
        
        # Tampilkan Hasil Evaluasi Metrik
        st.subheader("Hasil Evaluasi Data Testing:")
        res_df = pd.DataFrame([{
            'RMSE (Rp)': round(rmse, 2),
            'MAE (Rp)': round(mae, 2),
            'MAPE (%)': round(mape, 4)
        }])
        st.dataframe(res_df)

        # Plot Hasil Prediksi ke Layar Web
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true_plot, label='Harga Aktual', color='royalblue', linewidth=2)
        ax.plot(y_pred_plot, label='Harga Prediksi', color='crimson', linestyle='--', linewidth=2)
        ax.set_title(f"Perbandingan Harga Aktual vs Prediksi ({pilihan_optimasi})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
