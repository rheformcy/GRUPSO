# =========================================================
# IMPORT LIBRARY
# =========================================================
import os
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from pyswarms.single.global_best import GlobalBestPSO

# =========================================================
# DETERMINISM & REPRODUCIBILITY (PALING PENTING)
# =========================================================
os.environ["PYTHONHASHSEED"] = "49"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Pure CPU (ubah jadi "" jika mau GPU)

# Force single thread untuk reproducibility maksimal
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

SEED = 49
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="GRU-PSO Forecasting", layout="wide")
st.title("📈 Forecasting Harga Emas Menggunakan GRU-PSO")
st.markdown("Optimasi hyperparameter menggunakan **Particle Swarm Optimization (PSO)** pada model **GRU**.")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx", "xls"])

st.sidebar.divider()
st.sidebar.header("⚙️ Parameter")
timestep = st.sidebar.number_input("Timestep", min_value=1, max_value=30, value=5)
particles = st.sidebar.number_input("Jumlah Partikel", min_value=10, value=30)
iterasi = st.sidebar.number_input("Jumlah Iterasi", min_value=1, value=10)
epochs_pso = st.sidebar.number_input("Epoch PSO", min_value=5, value=10)
epochs_final = st.sidebar.number_input("Epoch Final", min_value=20, value=100)

st.sidebar.divider()
st.sidebar.header("🎛️ Range Hyperparameter")
units_min, units_max = st.sidebar.slider("Units", 16, 256, (32, 128))
lr_min = st.sidebar.number_input("Learning Rate Min", 0.0001, 0.01, 0.0005, format="%.5f")
lr_max = st.sidebar.number_input("Learning Rate Max", 0.0001, 0.05, 0.005, format="%.5f")
batch_min, batch_max = st.sidebar.slider("Batch Size", 8, 128, (16, 64))
dropout_min, dropout_max = st.sidebar.slider("Dropout", 0.0, 0.7, (0.1, 0.5), 0.05)

start_button = st.sidebar.button("🚀 Mulai Optimasi", use_container_width=True, type="primary")

# =========================================================
# MAIN
# =========================================================
if uploaded_file is None:
    st.info("📂 Silakan upload dataset terlebih dahulu.")
    st.stop()

# Read Data
df = pd.read_excel(uploaded_file)
df.columns = df.columns.str.strip()
df = df.replace(r'^\s*$', pd.NA, regex=True)
df.replace(["-", "?", "null", "NULL"], pd.NA, inplace=True)
df["Terakhir"] = pd.to_numeric(df["Terakhir"], errors="coerce")
df = df.dropna().reset_index(drop=True)

if start_button:
    # Reset semua seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    clear_session()
    gc.collect()

    st.subheader("🔄 Preprocessing...")
    # Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[["Terakhir"]])

    # Windowing
    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_seq, y_seq = create_sequences(data_scaled, timestep)

    # Split
    train_size = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_test, y_test = X_seq[train_size:], y_seq[train_size:]

    # Reshape
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Validation split
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    # =========================================================
    # PSO OBJECTIVE FUNCTION
    # =========================================================
    def objective_function(particles_array):
        n_particles = particles_array.shape[0]
        losses = np.zeros(n_particles)

        for i, particle in enumerate(particles_array):
            try:
                units = int(round(particle[0]))
                lr = float(particle[1])
                batch = int(round(particle[2]))
                dropout = float(particle[3])

                units = max(16, units)
                batch = max(8, batch)

                clear_session()
                tf.random.set_seed(SEED)

                model = Sequential([
                    Input(shape=(X_tr.shape[1], 1)),
                    GRU(units=units, 
                        activation='tanh',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=SEED)),
                    Dropout(dropout),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

                history = model.fit(
                    X_tr, y_tr,
                    epochs=epochs_pso,
                    batch_size=batch,
                    validation_data=(X_val, y_val),
                    verbose=0,
                    shuffle=False
                )

                losses[i] = history.history['val_loss'][-1]
                clear_session()
                gc.collect()

            except:
                losses[i] = 1e12

        return losses

    # =========================================================
    # RUN PSO
    # =========================================================
    st.write("🚀 Menjalankan Particle Swarm Optimization...")
    progress_bar = st.progress(0)

    bounds = ([units_min, lr_min, batch_min, dropout_min],
              [units_max, lr_max, batch_max, dropout_max])

    options = {'c1': 2.0, 'c2': 2.0, 'w': 0.7}

    optimizer = GlobalBestPSO(n_particles=particles, dimensions=4, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(objective_function, iters=iterasi, verbose=True)

    progress_bar.progress(50)

    # Best Parameters
    best_units = int(round(best_pos[0]))
    best_lr = float(best_pos[1])
    best_batch = int(round(best_pos[2]))
    best_dropout = float(best_pos[3])

    # =========================================================
    # FINAL MODEL
    # =========================================================
    st.write("🤖 Training Final Model...")
    clear_session()
    tf.random.set_seed(SEED)

    model_final = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        GRU(units=best_units,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=SEED)),
        Dropout(best_dropout),
        Dense(1)
    ])

    model_final.compile(optimizer=Adam(learning_rate=best_lr), loss='mse')

    history = model_final.fit(
        X_train, y_train,
        epochs=epochs_final,
        batch_size=best_batch,
        validation_split=0.2,
        shuffle=False,
        verbose=1
    )

    progress_bar.progress(100)

    # =========================================================
    # EVALUASI
    # =========================================================
    y_pred_scaled = model_final.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100

    # Tampilkan Hasil
    st.success("✅ Optimasi selesai!")

    st.subheader("🏆 Best Hyperparameter")
    st.dataframe(pd.DataFrame({
        "Units": [best_units],
        "Learning Rate": [best_lr],
        "Batch Size": [best_batch],
        "Dropout": [best_dropout]
    }), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:,.2f}")
    col2.metric("MAE", f"{mae:,.2f}")
    col3.metric("MAPE", f"{mape:.4f}%")

    # Plot
    st.subheader("📉 Training & Validation Loss")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Actual vs Prediction")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_actual, label="Actual", linewidth=2)
    ax2.plot(y_pred, label="Prediction", linewidth=2)
    ax2.legend()
    st.pyplot(fig2)
