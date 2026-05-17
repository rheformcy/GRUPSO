import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import gc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pyswarms.single.global_best import GlobalBestPSO

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="GRU-PSO Forecasting",
    layout="wide"
)

st.title("📈 Prediksi Harga Emas Menggunakan GRU-PSO")

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

# =========================================================
# SIDEBAR FORM
# =========================================================
with st.sidebar.form("form_pso"):

    st.header("⚙️ Pengaturan Model")

    # =====================================================
    # PSO PARAMETER
    # =====================================================
    particles = st.number_input(
        "Jumlah Partikel",
        min_value=5,
        max_value=100,
        value=40
    )

    iterasi = st.number_input(
        "Jumlah Iterasi PSO",
        min_value=1,
        max_value=100,
        value=5
    )

    epochs_pso = st.number_input(
        "Epoch PSO",
        min_value=1,
        max_value=100,
        value=10
    )

    epochs_final = st.number_input(
        "Epoch Final Model",
        min_value=1,
        max_value=500,
        value=100
    )

    # =====================================================
    # BOUND HYPERPARAMETER
    # =====================================================
    st.subheader("Bound Hyperparameter")

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

    # =====================================================
    # SUBMIT BUTTON
    # =====================================================
    submit_button = st.form_submit_button(
        "🚀 Jalankan GRU-PSO"
    )

# =========================================================
# START PROCESS
# =========================================================
if submit_button:
    st.session_state.running = True

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset Excel",
    type=["xlsx"]
)

# =========================================================
# MAIN PROCESS
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
    # VISUALISASI DATA
    # =====================================================
    st.subheader("📈 Visualisasi Harga Emas")

    fig_data, ax_data = plt.subplots(
        figsize=(12, 5)
    )

    ax_data.plot(df["Terakhir"])

    ax_data.set_title("Harga Emas")

    st.pyplot(fig_data)

    # =====================================================
    # SCALING
    # =====================================================
    feature_cols = ["Terakhir"]
    target_col = "Terakhir"

    data_features = df[feature_cols].values
    data_target = df[[target_col]].values

    values = df[["Terakhir"]].values

    n = len(values)

    n_train = int(n * 0.8)

    scaler_X = MinMaxScaler().fit(
        data_features[:n_train]
    )

    scaler_y = MinMaxScaler().fit(
        data_target[:n_train]
    )

    Xs = scaler_X.transform(data_features)

    ys = scaler_y.transform(data_target)

    # =====================================================
    # WINDOWING
    # =====================================================
    window = 1

    def make_sequences(
        X_scaled,
        y_scaled,
        window
    ):

        X_seq = []
        y_seq = []

        for i in range(window, len(X_scaled)):

            X_seq.append(
                X_scaled[i-window:i]
            )

            y_seq.append(
                y_scaled[i]
            )

        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = make_sequences(
        Xs,
        ys,
        window
    )

    dtrain_end = n_train - window

    X_train = X_seq_all[:dtrain_end]
    y_train = y_seq_all[:dtrain_end]

    X_test = X_seq_all[dtrain_end:]
    y_test = y_seq_all[dtrain_end:]

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
            method='ywm',
            ax=ax_pacf
        )

        st.pyplot(fig_pacf)

    # =====================================================
    # RUN MODEL
    # =====================================================
    if st.session_state.running:

        # =================================================
        # SET SEED
        # =================================================
        SEED = 49

        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

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
        def objective_function(particles):

            n_particles = particles.shape[0]

            losses = np.zeros(n_particles)

            for i, particle in enumerate(particles):

                units = int(
                    np.round(particle[0])
                )

                lr = float(particle[1])

                batch = int(
                    np.round(particle[2])
                )

                dropout = float(particle[3])

                try:

                    tf.random.set_seed(SEED)

                    clear_session()

                    model = Sequential([

                        Input(
                            shape=(
                                X_tr.shape[1],
                                X_tr.shape[2]
                            )
                        ),

                        GRU(
                            units=units,
                            activation='tanh'
                        ),

                        Dropout(dropout),

                        Dense(1)

                    ])

                    model.compile(
                        optimizer=Adam(
                            learning_rate=lr
                        ),
                        loss='mse'
                    )

                    model.fit(
                        X_tr,
                        y_tr,
                        epochs=epochs_pso,
                        batch_size=batch,
                        verbose=0,
                        shuffle=False
                    )

                    y_val_pred = model.predict(
                        X_val,
                        verbose=0
                    )

                    y_val_pred_inv = (
                        scaler_y.inverse_transform(
                            y_val_pred
                        ).flatten()
                    )

                    y_val_true_inv = (
                        scaler_y.inverse_transform(
                            y_val.reshape(-1, 1)
                        ).flatten()
                    )

                    losses[i] = mean_squared_error(
                        y_val_true_inv,
                        y_val_pred_inv
                    )

                except Exception as e:

                    st.write(
                        f"Error particle {i}: {e}"
                    )

                    losses[i] = 1e12

                clear_session()

                gc.collect()

            return losses

        # =================================================
        # PSO CONFIG
        # =================================================
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
            'c1': 2.0,
            'c2': 2.0,
            'w': 0.7
        }

        st.write("🚀 Menjalankan PSO...")

        progress_bar = st.progress(0)

        # =================================================
        # INIT PSO
        # =================================================
        optimizer = GlobalBestPSO(
            n_particles=particles,
            dimensions=4,
            options=options,
            bounds=bounds
        )

        n_particles, dims = (
            optimizer.swarm.position.shape
        )

        optimizer.swarm.pbest_pos = (
            optimizer.swarm.position.copy()
        )

        optimizer.swarm.pbest_cost = np.full(
            n_particles,
            np.inf
        )

        history_gbest_cost = []
        history_gbest_pos = []

        # =================================================
        # LOOP PSO MANUAL
        # =================================================
        for it in range(iterasi):

            costs = objective_function(
                optimizer.swarm.position
            )

            # =============================================
            # UPDATE PBEST
            # =============================================
            mask = (
                costs <
                optimizer.swarm.pbest_cost
            )

            optimizer.swarm.pbest_cost[mask] = (
                costs[mask]
            )

            optimizer.swarm.pbest_pos[mask] = (
                optimizer.swarm.position[mask].copy()
            )

            # =============================================
            # UPDATE GBEST
            # =============================================
            best_idx = np.argmin(
                optimizer.swarm.pbest_cost
            )

            optimizer.swarm.best_cost = (
                optimizer.swarm.pbest_cost[best_idx]
            )

            optimizer.swarm.best_pos = (
                optimizer.swarm.pbest_pos[best_idx].copy()
            )

            # =============================================
            # SAVE HISTORY
            # =============================================
            history_gbest_cost.append(
                float(
                    optimizer.swarm.best_cost
                )
            )

            history_gbest_pos.append(
                optimizer.swarm.best_pos.copy()
            )

            # =============================================
            # UPDATE VELOCITY
            # =============================================
            r1 = np.random.rand(
                *optimizer.swarm.position.shape
            )

            r2 = np.random.rand(
                *optimizer.swarm.position.shape
            )

            optimizer.swarm.velocity = (

                options['w']
                * optimizer.swarm.velocity

                + options['c1']
                * r1
                * (
                    optimizer.swarm.pbest_pos
                    - optimizer.swarm.position
                )

                + options['c2']
                * r2
                * (
                    optimizer.swarm.best_pos
                    - optimizer.swarm.position
                )
            )

            # =============================================
            # UPDATE POSITION
            # =============================================
            optimizer.swarm.position += (
                optimizer.swarm.velocity
            )

            lb = np.array(bounds[0])

            ub = np.array(bounds[1])

            optimizer.swarm.position = np.clip(
                optimizer.swarm.position,
                lb,
                ub
            )

            progress = int(
                ((it + 1) / iterasi) * 100
            )

            progress_bar.progress(progress)

            st.write(
                f"""
                Iterasi {it+1}

                Best Loss:
                {optimizer.swarm.best_cost:.6f}
                """
            )

        # =================================================
        # BEST PARAMETER
        # =================================================
        best_pos = history_gbest_pos[-1]

        best_cost = history_gbest_cost[-1]

        best_units = int(
            np.round(best_pos[0])
        )

        best_lr = float(best_pos[1])

        best_batch = int(
            np.round(best_pos[2])
        )

        best_dropout = float(best_pos[3])

        # =================================================
        # BEST HYPERPARAMETER
        # =================================================
        st.subheader("🏆 Best Hyperparameter")

        st.dataframe(pd.DataFrame({

            "Units": [best_units],

            "Learning Rate": [best_lr],

            "Batch Size": [best_batch],

            "Dropout": [best_dropout],

            "Best MSE": [best_cost]

        }))

        # =================================================
        # GRAFIK KONVERGENSI
        # =================================================
        st.subheader("📉 Grafik Konvergensi PSO")

        fig_conv, ax_conv = plt.subplots(
            figsize=(8, 5)
        )

        ax_conv.plot(
            range(
                1,
                len(history_gbest_cost)+1
            ),
            history_gbest_cost,
            marker='o'
        )

        ax_conv.set_xlabel("Iterasi")

        ax_conv.set_ylabel(
            "Global Best Loss"
        )

        ax_conv.set_title(
            "Konvergensi PSO"
        )

        ax_conv.grid(True)

        st.pyplot(fig_conv)

        # =================================================
        # FINAL MODEL
        # =================================================
        st.write("🤖 Training Final Model...")

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
                activation='tanh'
            ),

            Dropout(best_dropout),

            Dense(1)

        ])

        model_final.compile(

            optimizer=Adam(
                learning_rate=best_lr
            ),

            loss='mse'
        )

        history = model_final.fit(

            X_train,
            y_train,

            epochs=epochs_final,

            batch_size=best_batch,

            validation_split=0.2,

            verbose=1,

            shuffle=False
        )

        # =================================================
        # PREDICTION
        # =================================================
        y_pred_scaled = model_final.predict(
            X_test,
            verbose=0
        )

        y_pred = scaler_y.inverse_transform(
            y_pred_scaled
        ).flatten()

        y_actual = scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        # =================================================
        # EVALUATION
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
            history.history['loss'],
            label='Training Loss'
        )

        ax_loss.plot(
            history.history['val_loss'],
            label='Validation Loss'
        )

        ax_loss.legend()

        ax_loss.grid(True)

        st.pyplot(fig_loss)

        # =================================================
        # ACTUAL VS PREDICTION
        # =================================================
        st.subheader("📈 Actual vs Prediction")

        fig_pred, ax_pred = plt.subplots(
            figsize=(14, 7)
        )

        ax_pred.plot(
            y_actual,
            label='Actual',
            linewidth=2
        )

        ax_pred.plot(
            y_pred,
            label='Prediction',
            linestyle='--',
            linewidth=2
        )

        ax_pred.legend()

        ax_pred.grid(True)

        st.pyplot(fig_pred)

        # =================================================
        # STOP PROCESS
        # =================================================
        st.session_state.running = False
