import os
os.environ['PYTHONHASHSEED'] = '49'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================================================
# IMPORT LIBRARY
# =========================================================
import gc
import random

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.backend import clear_session

from pyswarms.single.global_best import GlobalBestPSO

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Optimasi GRU-PSO Harga Emas",
    layout="wide"
)

# =========================================================
# SEED
# =========================================================
SEED = 49

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# =========================================================
# HEADER
# =========================================================
st.title("📈 Optimasi GRU-PSO Harga Emas")

st.markdown("""
Aplikasi Optimasi Hyperparameter

Menggunakan:

- GRU (Gated Recurrent Unit)
- Particle Swarm Optimization (PSO)
""")

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Konfigurasi")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset Excel",
    type=["xlsx", "xls"]
)

window = st.sidebar.number_input(
    "Window / Timestep",
    min_value=1,
    max_value=30,
    value=1
)

particles = st.sidebar.number_input(
    "Jumlah Partikel",
    min_value=1,
    value=5
)

iters = st.sidebar.number_input(
    "Jumlah Iterasi PSO",
    min_value=1,
    value=3
)

epochs_final = st.sidebar.number_input(
    "Epoch Final Training",
    min_value=1,
    value=10
)

# =========================================================
# BUTTON MULAI
# =========================================================
start_button = st.button(
    "🚀 Mulai Optimasi",
    use_container_width=True
)

# =========================================================
# JIKA FILE ADA
# =========================================================
if uploaded_file is not None:

    # =====================================================
    # LOAD DATA
    # =====================================================
    emas = pd.read_excel(uploaded_file)

    emas.columns = emas.columns.str.strip()

    st.subheader("📄 Preview Dataset")

    st.dataframe(
        emas.head(),
        use_container_width=True
    )

    # =====================================================
    # MISSING VALUE
    # =====================================================
    st.subheader("🧩 Missing Value")

    missing_df = pd.DataFrame({
        "Kolom": emas.columns,
        "Jumlah Missing": emas.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    # =====================================================
    # VISUALISASI MISSING
    # =====================================================
    st.subheader("📊 Visualisasi Missing Value")

    fig_msno = plt.figure(figsize=(10, 4))

    msno.matrix(emas)

    st.pyplot(fig_msno)

    # =====================================================
    # OUTLIER
    # =====================================================
    st.subheader("🚨 Deteksi Outlier")

    fig_box, ax_box = plt.subplots(figsize=(10, 4))

    sns.boxplot(
        x=emas['Terakhir'],
        color='gold',
        ax=ax_box
    )

    st.pyplot(fig_box)

    Q1 = emas['Terakhir'].quantile(0.25)
    Q3 = emas['Terakhir'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = emas[
        (
            emas['Terakhir'] < lower_bound
        ) |
        (
            emas['Terakhir'] > upper_bound
        )
    ]

    st.write(f"Jumlah Outlier: {len(outliers)}")

    st.dataframe(
        outliers,
        use_container_width=True
    )

    # =====================================================
    # TIME SERIES PLOT
    # =====================================================
    st.subheader("📈 Time Series Plot")

    fig_ts, ax_ts = plt.subplots(figsize=(12, 5))

    ax_ts.plot(
        emas['Terakhir'],
        linewidth=2
    )

    ax_ts.grid(alpha=0.3)

    st.pyplot(fig_ts)

    # =====================================================
    # ACF PACF
    # =====================================================
    st.subheader("📊 ACF")

    fig_acf = plt.figure(figsize=(10, 4))

    plot_acf(
        emas['Terakhir'],
        lags=30
    )

    st.pyplot(fig_acf)

    st.subheader("📊 PACF")

    fig_pacf = plt.figure(figsize=(10, 4))

    plot_pacf(
        emas['Terakhir'],
        lags=30,
        method='ywm'
    )

    st.pyplot(fig_pacf)

    # =====================================================
    # BUTTON START
    # =====================================================
    if start_button:

        with st.spinner("Optimasi GRU-PSO sedang berjalan..."):

            clear_session()
            gc.collect()

            # =================================================
            # PREPROCESSING
            # =================================================
            values = emas[['Terakhir']].values.astype(float)

            n = len(values)

            n_train = int(n * 0.8)

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            scaler_X.fit(values[:n_train])
            scaler_y.fit(values[:n_train])

            Xs = scaler_X.transform(values)
            ys = scaler_y.transform(values)

            # =================================================
            # WINDOWING
            # =================================================
            def make_sequences(X_scaled, y_scaled, window):

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

            # =================================================
            # VALIDATION
            # =================================================
            val_size = 0.2

            n_tr_val = int(
                len(X_train) * (1 - val_size)
            )

            X_tr = X_train[:n_tr_val]
            y_tr = y_train[:n_tr_val]

            X_val = X_train[n_tr_val:]
            y_val = y_train[n_tr_val:]

            # =================================================
            # BOUNDS
            # =================================================
            bounds = (
                [16, 0.0001, 16, 0.1],
                [128, 0.01, 128, 0.5]
            )

            # =================================================
            # FITNESS FUNCTION
            # =================================================
            def make_pso_obj():

                def obj_fn(particles):

                    n_particles = particles.shape[0]

                    costs = np.zeros(n_particles)

                    for i, p in enumerate(particles):

                        units = int(np.round(p[0]))
                        lr = float(p[1])
                        batch = int(np.round(p[2]))
                        dropout = float(p[3])

                        try:

                            clear_session()

                            tf.random.set_seed(SEED)

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
                                epochs=10,
                                batch_size=batch,
                                verbose=0,
                                shuffle=False
                            )

                            pred = model.predict(
                                X_val,
                                verbose=0
                            )

                            pred_inv = scaler_y.inverse_transform(
                                pred
                            ).flatten()

                            actual_inv = scaler_y.inverse_transform(
                                y_val.reshape(-1, 1)
                            ).flatten()

                            mse = mean_squared_error(
                                actual_inv,
                                pred_inv
                            )

                            costs[i] = mse

                            clear_session()
                            gc.collect()

                        except Exception as e:

                            costs[i] = 1e12

                    return costs

                return obj_fn

            # =================================================
            # OPTIMIZER
            # =================================================
            optimizer = GlobalBestPSO(
                n_particles=particles,
                dimensions=4,
                options={
                    'c1': 2.0,
                    'c2': 2.0,
                    'w': 0.7
                },
                bounds=bounds
            )

            # =================================================
            # PROGRESS BAR
            # =================================================
            progress_bar = st.progress(0)

            status_text = st.empty()

            # =================================================
            # OPTIMASI
            # =================================================
            best_cost, best_pos = optimizer.optimize(
                make_pso_obj(),
                iters=iters
            )

            progress_bar.progress(100)

            status_text.success(
                "Optimasi selesai!"
            )

            # =================================================
            # BEST PARAMETER
            # =================================================
            best_units = int(np.round(best_pos[0]))
            best_lr = float(best_pos[1])
            best_batch = int(np.round(best_pos[2]))
            best_dropout = float(best_pos[3])

            # =================================================
            # FINAL MODEL
            # =================================================
            clear_session()

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
            # PREDIKSI
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
            # METRICS
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
            # BEST PARAMETER TABLE
            # =================================================
            st.subheader("🏆 Best Hyperparameter")

            best_df = pd.DataFrame({

                "Units": [best_units],
                "Learning Rate": [best_lr],
                "Batch Size": [best_batch],
                "Dropout": [best_dropout]

            })

            st.dataframe(
                best_df,
                use_container_width=True
            )

            # =================================================
            # METRICS
            # =================================================
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
            # KONVERGENSI
            # =================================================
            st.subheader("📉 Grafik Konvergensi PSO")

            fig1, ax1 = plt.subplots(figsize=(10, 5))

            ax1.plot(
                optimizer.cost_history,
                marker='o'
            )

            ax1.grid(alpha=0.3)

            st.pyplot(fig1)

            # =================================================
            # LOSS
            # =================================================
            st.subheader("📉 Training vs Validation Loss")

            fig2, ax2 = plt.subplots(figsize=(10, 5))

            ax2.plot(
                history.history['loss'],
                label='Training Loss'
            )

            ax2.plot(
                history.history['val_loss'],
                label='Validation Loss'
            )

            ax2.legend()

            ax2.grid(alpha=0.3)

            st.pyplot(fig2)

            # =================================================
            # AKTUAL VS PREDIKSI
            # =================================================
            st.subheader("📈 Aktual vs Prediksi")

            fig3, ax3 = plt.subplots(figsize=(12, 6))

            ax3.plot(
                y_actual,
                label='Aktual'
            )

            ax3.plot(
                y_pred,
                label='Prediksi'
            )

            ax3.legend()

            ax3.grid(alpha=0.3)

            st.pyplot(fig3)

            st.success(
                "Optimasi GRU-PSO berhasil!"
            )

else:

    st.info(
        "📂 Upload dataset Excel terlebih dahulu."
    )
