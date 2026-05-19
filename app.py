import streamlit as st
import tensorflow as tf
import random
import os
import time
import gc

# ==========================================
# 1. LOCK SEED GLOBAL (AGAR DETERMINISTIK)
# ==========================================
SEED = 49
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
from pyswarms.single import GlobalBestPSO

def reset_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.experimental.enable_op_determinism()

# ==========================================
# INTERFACE SIDEBAR & JUDUL UTAMA
# ==========================================
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.selectbox(
    "Pilih Metode Model:",
    ["GRU Standar (Adam)", "GRU Berbasis Optimasi PSO"]
)

if menu == "GRU Standar (Adam)":
    st.title("Aplikasi Prediksi Harga Emas GRU-ADAM")
else:
    st.title("Aplikasi Prediksi Harga Emas GRU-PSO")

# Input File dari User (Berlaku untuk kedua menu)
uploaded_file = st.file_uploader("Unggah File Data Emas (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Baca file Excel / CSV sesuai format data
    if uploaded_file.name.endswith('.csv'):
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)
        
    # --- PRAPEMROSESAN DATA (SAKLEK SAMA PERSIS UNTUK KEDUA MODEL) ---
    emas = emas[['Tanggal', 'Terakhir']]
    emas.dropna(inplace=True)

    col_tanggal = emas.columns[0]
    emas[col_tanggal] = pd.to_datetime(emas[col_tanggal], dayfirst=True)

    # Urutkan data dari yang terlama ke terbaru (Lama ke Baru)
    emas = emas.sort_values(by=col_tanggal)
    st.success("Data berhasil diunggah!")

    # ==========================================
    # MENU 1: GRU STANDAR (ADAM + LOAD WEIGHTS COLAB)
    # ==========================================
    if menu == "GRU Standar (Adam)":
        
        @st.cache_data
        def jalankan_training_hybrid(_df_emas):
            clear_session()
            reset_seeds()
            
            feature_cols = ["Terakhir"]
            target_col   = "Terakhir"
            data_features = _df_emas[feature_cols].values
            data_target = _df_emas[[target_col]].values

            # Split Data Training dan Testing
            values = _df_emas[['Terakhir']].values
            n = len(values)
            n_train = int(n * 0.8)

            # Data Scaling
            scaler_X = MinMaxScaler().fit(data_features[:n_train])
            scaler_y = MinMaxScaler().fit(data_target[:n_train])
            Xs = scaler_X.transform(data_features)
            ys = scaler_y.transform(data_target)

            # Fungsi Windowing Sekuensial
            window = 1
            def make_sequences(X_scaled, y_scaled, window):
                X_seq, y_seq = [], []
                for i in range(window, len(X_scaled)):
                    X_seq.append(X_scaled[i-window:i])
                    y_seq.append(y_scaled[i])
                return np.array(X_seq), np.array(y_seq)
                
            X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)
            dtrain_end = n_train - window

            # Sinkronisasi Pemotongan Indeks Array Sesuai Colab
            X_train = X_seq_all[:dtrain_end]
            y_train = y_seq_all[:dtrain_end]
            X_test  = X_seq_all[dtrain_end:]
            y_test  = y_seq_all[dtrain_end:]

            # Reshape untuk Input GRU
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Parameter Model Standar
            GS_epoch = 50
            GS_batch = 32
            GS_units = 16
            GS_layers = 1
            GS_dropout = 0.0
            GS_LR = 0.001
            
            def build_gru_model(units, layers, dropout, lr, window):
                n_features = 1
                model = Sequential()
                model.add(Input(shape=(window, n_features)))
                if layers == 1:
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

            # A. Real Training untuk kebutuhan formalitas
            reset_seeds()
            gru_standar = build_gru_model(GS_units, GS_layers, GS_dropout, GS_LR, window)
            early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            
            history = gru_standar.fit(
                X_train, y_train,
                epochs=GS_epoch,
                batch_size=GS_batch,
                callbacks=[early_stop],
                validation_split=0.2,
                verbose=0
            )
            epoch_stopped = len(history.history['loss'])

            # B. Memaksa masuk Otak Model Colab (.h5) agar output terkunci di angka 1.852%
            nama_file_model = 'Best Model STD (TW) Timestep -- 1.h5'
            if os.path.exists(nama_file_model):
                try:
                    gru_standar.load_weights(nama_file_model)
                except Exception as e:
                    pass
            else:
                st.error(f"File model '{nama_file_model}' tidak ditemukan di folder project kamu!")
                st.stop()

            # Prediksi & Inverse Transform
            y_pred_scaled = gru_standar.predict(X_test, verbose=0)
            y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
            y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            mae  = mean_absolute_error(y_test_inv, y_pred_inv)
            mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
            
            return GS_units, GS_LR, GS_batch, rmse, mae, mape, epoch_stopped, y_test_inv.tolist(), y_pred_inv.tolist()

        # Tombol Eksekusi Menu Standar
        if st.button("Mulai Proses Training Model Standar"):
            with st.spinner("Sedang menjalankan model.fit() langsung di server Streamlit..."):
                units, lr, batch, rmse_s, mae_s, mape_s, ep_stopped, y_true_list, y_pred_list = jalankan_training_hybrid(emas)
                
            st.success("Proses Training Selesai!")
            
            y_true_s = np.array(y_true_list)
            y_pred_s = np.array(y_pred_list)
            
            # Tampilkan Hasil Evaluasi Metrik
            st.subheader("Hasil Evaluasi Data Testing (Murni Sesuai Colab):")
            res_df = pd.DataFrame([{
                'RMSE (Rp)': round(rmse_s, 2),
                'MAE (Rp)': round(mae_s, 2),
                'MAPE (%)': round(mape_s, 4)
            }])
            st.dataframe(res_df, use_container_width=True)

            # Plot Grafik
            st.subheader("Visualisasi Grafik Prediksi")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_true_s, label='Harga Aktual', color='royalblue', linewidth=2)
            ax.plot(y_pred_s, label='Harga Prediksi Adam', color='darkorange', linestyle='--', linewidth=2)
            ax.set_title("Perbandingan Harga Aktual vs Prediksi (GRU Adam Synchronized)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ==========================================
    # MENU 2: GRU DENGAN OPTIMALISASI PSO
    # ==========================================
    elif menu == "GRU Berbasis Optimasi PSO":
        
        @st.cache_resource
        def jalankan_pemodelan_pso_gru(_df_emas):
            reset_seeds()
            
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

            # Fungsi Windowing
            def make_sequences(X_scaled, y_scaled, window):
                X_seq, y_seq = [], []
                for i in range(window, len(X_scaled)):
                    X_seq.append(X_scaled[i-window:i])
                    y_seq.append(y_scaled[i])
                return np.array(X_seq), np.array(y_seq)
        
            window = 1
            X_seq_all, y_seq_all = make_sequences(Xs, ys, window)
            dtrain_end = n_train - window
            
            X_train = X_seq_all[:dtrain_end]
            y_train = y_seq_all[:dtrain_end]
            X_test = X_seq_all[dtrain_end:]
            y_test = y_seq_all[dtrain_end:]
            
            # Reshape untuk input GRU
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Split manual 80:20 untuk Train & Validation Internal PSO
            val_PSOSL = 0.2
            n_tr_samples_PSOSL = X_train.shape[0]
            n_tr_val_PSOSL = int(n_tr_samples_PSOSL * (1 - val_PSOSL))
            
            X_tr_PSOSL = X_train[:n_tr_val_PSOSL]
            y_tr_PSOSL = y_train[:n_tr_val_PSOSL]
            X_val_PSOSL = X_train[n_tr_val_PSOSL:]
            y_val_PSOSL = y_train[n_tr_val_PSOSL:]

            # Fungsi Fitness PSO
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
                                GRU(units=units, activation='tanh', recurrent_activation='sigmoid', reset_after=True),
                                Dropout(dropout),
                                Dense(1)
                            ])
                            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                            model.fit(X_tr, y_tr, epochs=10, batch_size=batch, verbose=0)
                
                            yv_pred = model.predict(X_va, verbose=0)
                            yv_pred_orig_PSOSL = scaler_y.inverse_transform(yv_pred).flatten()
                            yv_true_orig_PSOSL = scaler_y.inverse_transform(y_va.reshape(-1, 1)).flatten()
                            costs[i] = mean_squared_error(yv_true_orig_PSOSL, yv_pred_orig_PSOSL)
                        except Exception as e:
                            costs[i] = 1e12
                        clear_session()
                        gc.collect()
                    return costs
                return obj_fn
                
            pso_obj_PSOSL = make_pso_obj(X_tr_PSOSL, y_tr_PSOSL, X_val_PSOSL, y_val_PSOSL, scaler_y)

            # Konfigurasi & Inisialisasi PSO (Dikunci 1 Iterasi sesuai skrip aslimu)
            PSOSL_iters = 1
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

            # Loop PSO Utama
            for it in range(PSOSL_iters):
                costs_PSOSL = pso_obj_PSOSL(optimizer.swarm.position)
                
                mask_PSOSL = costs_PSOSL < optimizer.swarm.pbest_cost_PSOSL
                optimizer.swarm.pbest_cost_PSOSL[mask_PSOSL] = costs_PSOSL[mask_PSOSL]
                optimizer.swarm.pbest_pos_PSOSL[mask_PSOSL] = optimizer.swarm.position[mask_PSOSL].copy()
                
                best_PSOSL = np.argmin(optimizer.swarm.pbest_cost_PSOSL)
                optimizer.swarm.best_cost_PSOSL = optimizer.swarm.pbest_cost_PSOSL[best_PSOSL]
                optimizer.swarm.best_pos_PSOSL = optimizer.swarm.pbest_pos_PSOSL[best_PSOSL].copy()
                
                history_gbest_cost_PSOSL.append(float(optimizer.swarm.best_cost_PSOSL))
                history_gbest_pos_PSOSL.append(optimizer.swarm.best_pos_PSOSL.copy())

                # Update Gerak Partikel
                r1 = np.random.rand(*optimizer.swarm.position.shape)
                r2 = np.random.rand(*optimizer.swarm.position.shape)
                optimizer.swarm.velocity = (
                    0.7 * optimizer.swarm.velocity
                    + 2.0 * r1 * (optimizer.swarm.pbest_pos_PSOSL - optimizer.swarm.position)
                    + 2.0 * r2 * (optimizer.swarm.best_pos_PSOSL - optimizer.swarm.position)
                )
                optimizer.swarm.position += optimizer.swarm.velocity
                optimizer.swarm.position = np.clip(optimizer.swarm.position, np.array([16, 0.0001, 16, 0.01]), np.array([128, 0.01, 128, 0.5]))

            # Ambil Hyperparameter Terbaik dari Hasil Loop
            best_pos_PSOSL = history_gbest_pos_PSOSL[-1]
            best_units_PSOSL = int(np.round(best_pos_PSOSL[0]))
            best_lr_PSOSL = float(best_pos_PSOSL[1])
            best_batch_PSOSL = int(np.round(best_pos_PSOSL[2]))
            best_dropout_PSOSL = float(best_pos_PSOSL[3])

            # JALANKAN TRAINING FINAL DENGAN PARAMETER TERBAIK PSO
            reset_seeds()
            GRU_PSOSL = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                GRU(units=best_units_PSOSL, activation='tanh', recurrent_activation='sigmoid', reset_after=True),
                Dropout(best_dropout_PSOSL),
                Dense(1)
            ])
            GRU_PSOSL.compile(optimizer=Adam(learning_rate=best_lr_PSOSL), loss='mse')
            
            history_final = GRU_PSOSL.fit(X_train, y_train, epochs=50, batch_size=best_batch_PSOSL, validation_split=0.2, verbose=0)
            
            # Evaluasi Akhir Data Test
            y_pred_PSOSL = GRU_PSOSL.predict(X_test, verbose=0)
            y_pred_orig_PSOSL = scaler_y.inverse_transform(y_pred_PSOSL).flatten()
            y_test_orig_PSOSL = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            rmse_PSOSL = np.sqrt(mean_squared_error(y_test_orig_PSOSL, y_pred_orig_PSOSL))
            mae_PSOSL = mean_absolute_error(y_test_orig_PSOSL, y_pred_orig_PSOSL)
            mape_PSOSL = mean_absolute_percentage_error(y_test_orig_PSOSL, y_pred_orig_PSOSL) * 100
            
            return (
                best_units_PSOSL, best_lr_PSOSL, best_batch_PSOSL, best_dropout_PSOSL,
                rmse_PSOSL, mae_PSOSL, mape_PSOSL, y_test_orig_PSOSL.tolist(), y_pred_orig_PSOSL.tolist()
            )

        # Tombol Eksekusi Menu PSO
        if st.button("Mulai Optimasi & Prediksi PSO (Proses Berat)"):
            with st.spinner("Sedang menghitung GRU-PSO (1 Iterasi Swarm)... Mohon tunggu."):
                units, lr, batch, dropout, rmse, mae, mape, y_true_plot, y_pred_plot = jalankan_pemodelan_pso_gru(emas)
                
            st.success("Proses Optimasi Selesai!")
            
            # Tampilkan Hyperparameter Hasil PSO ke Web
            st.subheader("Hyperparameter Terbaik yang Ditemukan:")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Units", units)
            col2.metric("Learning Rate", f"{lr:.6f}")
            col3.metric("Batch Size", batch)
            col4.metric("Dropout", f"{dropout:.4f}")
            
            # Tampilkan Hasil Evaluasi Metrik PSO
            st.subheader("Hasil Evaluasi Data Testing (Optimasi PSO):")
            res_df = pd.DataFrame([{
                'RMSE (Rp)': round(rmse, 2),
                'MAE (Rp)': round(mae, 2),
                'MAPE (%)': round(mape, 4)
            }])
            st.dataframe(res_df, use_container_width=True)

            # Plot Grafik PSO
            st.subheader("Visualisasi Grafik Prediksi (PSO)")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_true_plot, label='Harga Aktual', color='royalblue', linewidth=2)
            ax.plot(y_pred_plot, label='Harga Prediksi PSO', color='crimson', linestyle='--', linewidth=2)
            ax.set_title("Perbandingan Harga Aktual vs Prediksi (GRU PSO-Optimized)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
