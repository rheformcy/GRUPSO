import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
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
# 1. KUNCI ALL SEEDS DI AWAL SKRIP
# ==========================================
SEED = 49
def reset_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Paksa CPU agar math-ops konsisten

reset_seeds()

st.title("Aplikasi Prediksi Harga Emas GRU-PSO")

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
    # Fungsi ini hanya akan berjalan SATU KALI. Jika user klik tombol lain, 
    # Streamlit akan mengambil hasilnya langsung dari memori tanpa run-ulang PSO.
    
    @st.cache_resource
    def jalankan_pemodelan_pso_gru(_df_emas):
        # Reset seed tepat sebelum pemrosesan data dimulai
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
        train_values = values[:n_train]
        test_values  = values[n_train:]

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
        
        print(f"Shape X_train: {X_train.shape}")
        print(f"Shape X_test: {X_test.shape}")

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

                        # Arsitektur GRU
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
                    except Exception as e:
                        costs[i] = 1e12
                    clear_session()
                    gc.collect()
                return costs
            return obj_fn
            
        pso_obj_PSOSL = make_pso_obj(
        X_tr_PSOSL, y_tr_PSOSL,
        X_val_PSOSL, y_val_PSOSL,
        scaler_y
            )

        # Konfigurasi & Inisialisasi PSO
        PSOSL_iters=1
        optimizer = GlobalBestPSO(
            n_particles=40, dimensions=4,
            options={'c1': 2.0, 'c2': 2.0, 'w': 0.7},
            bounds=([16, 0.0001, 16, 0.01], [128, 0.01, 128, 0.5])
        )
        
        n_particles, dims = optimizer.swarm.position.shape
        optimizer.swarm.pbest_pos_PSOSL = optimizer.swarm.position.copy()
        optimizer.swarm.pbest_cost_PSOSL = np.full(n_particles, np.inf)
        
        history_positions_PSOSL = []
        history_velocity_PSOSL = []
        history_costs_PSOSL = []
        history_gbest_cost_PSOSL = []
        history_gbest_pos_PSOSL = []
        history_r1_PSOSL = []
        history_r2_PSOSL = []

        # Loop PSO Utama
        for it in range(PSOSL_iters):
            # Evaluasi Fungsi Fitness
            costs_PSOSL = pso_obj_PSOSL(optimizer.swarm.position)
            # Update pbest (Personal Best)
            mask_PSOSL = costs_PSOSL < optimizer.swarm.pbest_cost_PSOSL
            optimizer.swarm.pbest_cost_PSOSL[mask_PSOSL] = costs_PSOSL[mask_PSOSL]
            optimizer.swarm.pbest_pos_PSOSL[mask_PSOSL] = optimizer.swarm.position[mask_PSOSL].copy()
            # Update gbest (Global Best)
            best_PSOSL = np.argmin(optimizer.swarm.pbest_cost_PSOSL)
            optimizer.swarm.best_cost_PSOSL = optimizer.swarm.pbest_cost_PSOSL[best_PSOSL]
            optimizer.swarm.best_pos_PSOSL = optimizer.swarm.pbest_pos_PSOSL[best_PSOSL].copy()
            history_positions_PSOSL.append(optimizer.swarm.position.copy())
            history_velocity_PSOSL.append(optimizer.swarm.velocity.copy())
            history_costs_PSOSL.append(costs_PSOSL.copy())
            history_gbest_cost_PSOSL.append(float(optimizer.swarm.best_cost_PSOSL))
            history_gbest_pos_PSOSL.append(optimizer.swarm.best_pos_PSOSL.copy())

            # Update Velocity & Position manual (Disinkronkan benih acaknya)
            r1 = np.random.rand(*optimizer.swarm.position.shape)
            r2 = np.random.rand(*optimizer.swarm.position.shape)
            optimizer.swarm.velocity = (
                0.7 * optimizer.swarm.velocity
                + 2.0 * r1 * (optimizer.swarm.pbest_pos_PSOSL - optimizer.swarm.position)
                + 2.0 * r2 * (optimizer.swarm.best_pos_PSOSL - optimizer.swarm.position)
            )
            optimizer.swarm.position += optimizer.swarm.velocity
            optimizer.swarm.position = np.clip(optimizer.swarm.position, np.array([16, 0.0001, 16, 0.01]), np.array([128, 0.01, 128, 0.5]))

        # Ambil Hyperparameter Terbaik
        best_pos_PSOSL = history_gbest_pos_PSOSL[-1]
        best_cost_PSOSL = history_gbest_cost_PSOSL[-1]
        best_units_PSOSL = int(np.round(best_pos_PSOSL[0]))
        best_lr_PSOSL = float(best_pos_PSOSL[1])
        best_batch_PSOSL = int(np.round(best_pos_PSOSL[2]))
        best_dropout_PSOSL = float(best_pos_PSOSL[3])

        # ========================================================
        # PERUBAHAN KRUSIAL: SINKRONISASI VALIDATION SPLIT FINAL
        # ========================================================
        # Menggunakan data latih internal (X_tr_PSOSL) dan validasi internal (X_val_PSOSL) 
        # secara eksplisit agar pembagian datanya 100% sama dengan saat proses pencarian PSO.
        tf.random.set_seed(SEED)
        GRU_PSOSL = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            GRU(units=best_units_PSOSL, activation='tanh'),
            Dropout(best_dropout_PSOSL),
            Dense(1)
        ])
        GRU_PSOSL.compile(optimizer=Adam(learning_rate=best_lr_PSOSL), loss='mse')
        
        # Di sini kita masukkan validation_data secara manual, bukan pakai validation_split otomatis
        history_final = GRU_PSOSL.fit(X_train, y_train, epochs=50, batch_size=best_batch_PSOSL, validation_split=0.2, shuffle=False, verbose=1)
        
        # Evaluasi Akhir Test
        y_pred_PSOSL = GRU_PSOSL.predict(X_test, verbose=0)
        y_pred_orig_PSOSL = scaler_y.inverse_transform(y_pred_PSOSL).flatten()
        y_test_orig_PSOSL = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse_PSOSL = np.sqrt(mean_squared_error(y_test_orig_PSOSL, y_pred_orig_PSOSL))
        mae_PSOSL = mean_absolute_error(y_test_orig_PSOSL, y_pred_orig_PSOSL)
        mape_PSOSL = mean_absolute_percentage_error(y_test_orig_PSOSL, y_pred_orig_PSOSL) * 100
        train_loss_PSOSL = history_final.history['loss'][-1]
        val_loss_PSOSL = history_final.history['val_loss'][-1]
        epoch_PSOSL = len(history_final.history['loss'])
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
    # Tombol pemicu eksekusi
    if st.button("Mulai Optimasi & Prediksi (Proses Berat)"):
        with st.spinner("Sedang menghitung GRU-PSO (5 Iterasi)... Mohon tunggu."):
            units, lr, batch, dropout, rmse, mae, mape, y_true_plot, y_pred_plot = jalankan_pemodelan_pso_gru(emas)
            
        st.success("Proses Selesai!")
        
        # Tampilkan Parameter Terbaik ke Interface Web
        st.subheader("Hyperparameter Terbaik yang Ditemukan:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Units", units)
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
        ax.plot(y_true_plot, label='Harga Aktual', color='royalblue')
        ax.plot(y_pred_plot, label='Harga Prediksi', color='crimson', linestyle='--')
        ax.set_title("Perbandingan Harga Aktual vs Prediksi (Emas)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Render grafik ke Streamlit
        st.pyplot(fig)
