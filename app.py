import os

# =========================================================
# REPRODUCIBILITY
# =========================================================
os.environ["PYTHONHASHSEED"] = "49"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from pyswarms.single.global_best import (
    GlobalBestPSO
)

# =========================================================
# FIXED SEED
# =========================================================
SEED = 49

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="GRU-PSO Forecasting",
    layout="wide"
)

st.title("📈 Prediksi Harga Emas Menggunakan GRU-PSO")

st.markdown("""
Optimasi hyperparameter menggunakan
Particle Swarm Optimization (PSO)
pada model GRU.
""")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar.form("form_pso"):

    st.header("⚙️ Parameter Model")

    timestep = st.number_input(
        "Timestep",
        min_value=1,
        max_value=30,
        value=5
    )

    particles = st.number_input(
        "Jumlah Partikel",
        min_value=5,
        max_value=100,
        value=30
    )

    iterasi = st.number_input(
        "Jumlah Iterasi PSO",
        min_value=1,
        max_value=100,
        value=10
    )

    epochs_pso = st.number_input(
        "Epoch PSO",
        min_value=1,
        max_value=100,
        value=10
    )

    epochs_final = st.number_input(
        "Epoch Final",
        min_value=1,
        max_value=500,
        value=100
    )

    st.subheader("🎛️ Bound Hyperparameter")

    units_min = st.number_input(
        "Units Min",
        value=16
    )

    units_max = st.number_input(
        "Units Max",
        value=128
    )

    lr_min = st.number_input(
        "Learning Rate Min",
        value=0.0001,
        format="%.4f"
    )

    lr_max = st.number_input(
        "Learning Rate Max",
        value=0.01,
        format="%.4f"
    )

    batch_min = st.number_input(
        "Batch Size Min",
        value=16
    )

    batch_max = st.number_input(
        "Batch Size Max",
        value=128
    )

    dropout_min = st.number_input(
        "Dropout Min",
        value=0.1
    )

    dropout_max = st.number_input(
        "Dropout Max",
        value=0.5
    )

    run_model = st.form_submit_button(
        "🚀 Jalankan GRU-PSO"
    )

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset Excel",
    type=["xlsx", "xls"]
)

# =========================================================
# MAIN
# =========================================================
if uploaded_file is not None:

    # =====================================================
    # READ DATA
    # =====================================================
    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    df = df.replace(
        r'^\s*$',
        pd.NA,
        regex=True
    )

    df.replace(
        ["-", "?", "null", "NULL"],
        pd.NA,
        inplace=True
    )

    df["Terakhir"] = pd.to_numeric(
        df["Terakhir"],
        errors="coerce"
    )

    df = df.dropna().reset_index(drop=True)

    # =====================================================
    # DATASET
    # =====================================================
    st.subheader("📋 Dataset")

    st.dataframe(df.head())

    # =====================================================
    # VISUALISASI
    # =====================================================
    st.subheader("📈 Visualisasi Harga Emas")

    fig_data, ax_data = plt.subplots(
        figsize=(12, 5)
    )

    ax_data.plot(
        df["Terakhir"],
        linewidth=2
    )

    ax_data.set_title("Harga Emas")

    ax_data.grid(True)

    st.pyplot(fig_data)

    # =====================================================
    # ACF PACF
    # =====================================================
    st.subheader("📊 ACF & PACF")

    col1, col2 = st.columns(2)

    with col1:

        fig_acf, ax_acf = plt.subplots(
            figsize=(6, 4)
        )

        plot_acf(
            df["Terakhir"],
            lags=30,
            ax=ax_acf
        )

        st.pyplot(fig_acf)

    with col2:

        fig_pacf, ax_pacf = plt.subplots(
            figsize=(6, 4)
        )

        plot_pacf(
            df["Terakhir"],
            lags=30,
            method="ywm",
            ax=ax_pacf
        )

        st.pyplot(fig_pacf)

    # =====================================================
    # RUN MODEL
    # =====================================================
    if run_model:

        # =================================================
        # RESET SEED
        # =================================================
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        tf.keras.utils.set_random_seed(SEED)

        clear_session()
        gc.collect()

        # =================================================
        # TRAIN TEST SPLIT
        # =================================================
        values = df[["Terakhir"]].values

        n = len(values)

        n_train = int(n * 0.8)

        train_data = values[:n_train]

        test_data = values[n_train:]

        # =================================================
        # SCALER FIT HANYA TRAIN
        # =================================================
        scaler = MinMaxScaler()

        scaler.fit(train_data)

        train_scaled = scaler.transform(
            train_data
        )

        test_scaled = scaler.transform(
            test_data
        )

        # =================================================
        # GABUNG KEMBALI
        # =================================================
        scaled_all = np.concatenate([
            train_scaled,
            test_scaled
        ])

        # =================================================
        # WINDOWING
        # =================================================
        def make_sequences(data, window):

            X = []
            y = []

            for i in range(window, len(data)):

                X.append(
                    data[i-window:i]
                )

                y.append(
                    data[i]
                )

            return np.array(X), np.array(y)

        X_all, y_all = make_sequences(
            scaled_all,
            timestep
        )

        # =================================================
        # SPLIT WINDOW
        # =================================================
        train_end = n_train - timestep

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]

        X_test = X_all[train_end:]
        y_test = y_all[train_end:]

        # =================================================
        # RESHAPE
        # =================================================
        X_train = X_train.reshape(
            (
                X_train.shape[0],
                X_train.shape[1],
                1
            )
        )

        X_test = X_test.reshape(
            (
                X_test.shape[0],
                X_test.shape[1],
                1
            )
        )

        # =================================================
        # VALIDATION SPLIT
        # =================================================
        val_size = 0.2

        n_train_samples = X_train.shape[0]

        n_train_val = int(
            n_train_samples * (1 - val_size)
        )

        X_tr = X_train[:n_train_val]
        y_tr = y_train[:n_train_val]

        X_val = X_train[n_train_val:]
        y_val = y_train[n_train_val:]

        # =================================================
        # OBJECTIVE FUNCTION
        # =================================================
        def objective_function(
            particles_array
        ):

            n_particles = (
                particles_array.shape[0]
            )

            losses = np.zeros(
                n_particles
            )

            for i, particle in enumerate(
                particles_array
            ):

                try:

                    units = int(
                        np.round(
                            particle[0]
                        )
                    )

                    lr = float(
                        particle[1]
                    )

                    batch = int(
                        np.round(
                            particle[2]
                        )
                    )

                    dropout = float(
                        particle[3]
                    )

                    clear_session()

                    tf.random.set_seed(
                        SEED
                    )

                    model = Sequential([

                        Input(
                            shape=(
                                X_tr.shape[1],
                                X_tr.shape[2]
                            )
                        ),

                        GRU(
                            units=units,
                            activation="tanh",

                            kernel_initializer=tf.keras.initializers.GlorotUniform(
                                seed=SEED
                            ),

                            recurrent_initializer=tf.keras.initializers.Orthogonal(
                                seed=SEED
                            )
                        ),

                        Dropout(dropout),

                        Dense(1)

                    ])

                    model.compile(

                        optimizer=Adam(
                            learning_rate=lr
                        ),

                        loss="mse"
                    )

                    history = model.fit(

                        X_tr,
                        y_tr,

                        validation_data=(
                            X_val,
                            y_val
                        ),

                        epochs=epochs_pso,

                        batch_size=batch,

                        verbose=0,

                        shuffle=False
                    )

                    losses[i] = (
                        history.history[
                            "val_loss"
                        ][-1]
                    )

                except Exception:

                    losses[i] = 1e12

                clear_session()

                gc.collect()

            return losses

        # =================================================
        # PSO
        # =================================================
        st.write(
            "🚀 Menjalankan PSO..."
        )

        progress_bar = st.progress(0)

        bounds = (

            [
                units_min,
                lr_min,
                batch_min,
                dropout_min
            ],

            [
                units_max,
                lr_max,
                batch_max,
                dropout_max
            ]
        )

        options = {
            "c1": 2.0,
            "c2": 2.0,
            "w": 0.7
        }

        optimizer = GlobalBestPSO(

            n_particles=particles,

            dimensions=4,

            options=options,

            bounds=bounds
        )

        best_cost, best_pos = (

            optimizer.optimize(

                objective_function,

                iters=iterasi,

                verbose=False
            )
        )

        progress_bar.progress(50)

        # =================================================
        # BEST PARAMETER
        # =================================================
        best_units = int(
            np.round(best_pos[0])
        )

        best_lr = float(best_pos[1])

        best_batch = int(
            np.round(best_pos[2])
        )

        best_dropout = float(best_pos[3])

        st.subheader(
            "🏆 Best Hyperparameter"
        )

        st.dataframe(pd.DataFrame({

            "Units": [best_units],

            "Learning Rate": [best_lr],

            "Batch Size": [best_batch],

            "Dropout": [best_dropout],

            "Best Loss": [best_cost]

        }))

        # =================================================
        # FINAL MODEL
        # =================================================
        st.write(
            "🤖 Training Final Model..."
        )

        clear_session()

        tf.random.set_seed(SEED)

        model_final = Sequential([

            Input(
                shape=(
                    X_train.shape[1],
                    X_train.shape[2]
                )
            ),

            GRU(
                units=best_units,
                activation="tanh",

                kernel_initializer=tf.keras.initializers.GlorotUniform(
                    seed=SEED
                ),

                recurrent_initializer=tf.keras.initializers.Orthogonal(
                    seed=SEED
                )
            ),

            Dropout(best_dropout),

            Dense(1)

        ])

        model_final.compile(

            optimizer=Adam(
                learning_rate=best_lr
            ),

            loss="mse"
        )

        history = model_final.fit(

            X_train,
            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=epochs_final,

            batch_size=best_batch,

            verbose=1,

            shuffle=False
        )

        progress_bar.progress(100)

        # =================================================
        # PREDICTION
        # =================================================
        y_pred_scaled = model_final.predict(
            X_test,
            verbose=0
        )

        y_pred = scaler.inverse_transform(
            y_pred_scaled
        ).flatten()

        y_actual = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        # =================================================
        # EVALUASI
        # =================================================
        rmse = np.sqrt(
            mean_squared_error(
                y_actual,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_actual,
            y_pred
        )

        mape = (
            mean_absolute_percentage_error(
                y_actual,
                y_pred
            ) * 100
        )

        # =================================================
        # METRICS
        # =================================================
        st.subheader("📊 Evaluasi Model")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "RMSE",
            f"{rmse:,.2f}"
        )

        col2.metric(
            "MAE",
            f"{mae:,.2f}"
        )

        col3.metric(
            "MAPE",
            f"{mape:.4f}%"
        )

        # =================================================
        # LOSS PLOT
        # =================================================
        st.subheader(
            "📉 Training & Validation Loss"
        )

        fig_loss, ax_loss = plt.subplots(
            figsize=(10, 5)
        )

        ax_loss.plot(
            history.history["loss"],
            label="Training Loss"
        )

        ax_loss.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        ax_loss.legend()

        ax_loss.grid(True)

        st.pyplot(fig_loss)

        # =================================================
        # ACTUAL VS PREDICTION
        # =================================================
        st.subheader(
            "📈 Actual vs Prediction"
        )

        fig_pred, ax_pred = plt.subplots(
            figsize=(14, 7)
        )

        ax_pred.plot(
            y_actual,
            label="Actual",
            linewidth=2
        )

        ax_pred.plot(
            y_pred,
            label="Prediction",
            linestyle="--",
            linewidth=2
        )

        ax_pred.legend()

        ax_pred.grid(True)

        st.pyplot(fig_pred)

        st.success(
            "✅ Proses selesai!"
        )
