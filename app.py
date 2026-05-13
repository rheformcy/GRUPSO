# =========================================================
# IMPORT LIBRARY
# =========================================================
import os

os.environ["PYTHONHASHSEED"] = "49"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

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
# SEED
# =========================================================
SEED = 49

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="GRU-PSO Forecasting",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================
st.title("📈 Forecasting Harga Emas Menggunakan GRU-PSO")

st.markdown("""
Optimasi hyperparameter menggunakan
**Particle Swarm Optimization (PSO)**
pada model
**Gated Recurrent Unit (GRU)**.
""")

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

st.sidebar.divider()

# =========================================================
# PARAMETER
# =========================================================
st.sidebar.header("⚙️ Parameter")

timestep = st.sidebar.number_input(
    "Timestep",
    min_value=1,
    max_value=30,
    value=1
)

particles = st.sidebar.number_input(
    "Jumlah Partikel",
    min_value=1,
    value=2
)

iterasi = st.sidebar.number_input(
    "Jumlah Iterasi",
    min_value=1,
    value=1
)

epochs_pso = st.sidebar.number_input(
    "Epoch PSO",
    min_value=1,
    value=1
)

epochs_final = st.sidebar.number_input(
    "Epoch Final",
    min_value=1,
    value=5
)

# =========================================================
# RANGE HYPERPARAMETER
# =========================================================
st.sidebar.divider()

st.sidebar.header("🎛️ Range Hyperparameter")

units_min = st.sidebar.number_input(
    "Units Min",
    min_value=1,
    value=16
)

units_max = st.sidebar.number_input(
    "Units Max",
    min_value=1,
    value=128
)

lr_min = st.sidebar.number_input(
    "Learning Rate Min",
    min_value=0.0001,
    value=0.0001,
    format="%.4f"
)

lr_max = st.sidebar.number_input(
    "Learning Rate Max",
    min_value=0.0001,
    value=0.01,
    format="%.4f"
)

batch_min = st.sidebar.number_input(
    "Batch Size Min",
    min_value=1,
    value=16
)

batch_max = st.sidebar.number_input(
    "Batch Size Max",
    min_value=1,
    value=128
)

dropout_min = st.sidebar.slider(
    "Dropout Min",
    min_value=0.0,
    max_value=0.9,
    value=0.1,
    step=0.1
)

dropout_max = st.sidebar.slider(
    "Dropout Max",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.1
)

# =========================================================
# BUTTON
# =========================================================
st.sidebar.divider()

start_button = st.sidebar.button(
    "🚀 Mulai Optimasi",
    use_container_width=True
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

    # =====================================================
    # PREVIEW
    # =====================================================
    st.subheader("📄 Preview Dataset")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # =====================================================
    # MISSING VALUE
    # =====================================================
    st.subheader("🧩 Missing Values")

    missing_df = pd.DataFrame({
        "Kolom": df.columns,
        "Jumlah Missing": df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    # =====================================================
    # OUTLIER
    # =====================================================
    st.subheader("🚨 Outlier")

    if "Terakhir" in df.columns:

        Q1 = df["Terakhir"].quantile(0.25)
        Q3 = df["Terakhir"].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df["Terakhir"] < lower) |
            (df["Terakhir"] > upper)
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

    if (
        "Tanggal" in df.columns and
        "Terakhir" in df.columns
    ):

        fig, ax = plt.subplots(figsize=(12, 5))

        tanggal = pd.to_datetime(df["Tanggal"])

        ax.plot(
            tanggal,
            df["Terakhir"],
            linewidth=2
        )

        ax.set_xlabel("Tahun")
        ax.set_ylabel("Harga")

        ax.grid(alpha=0.3)

        st.pyplot(fig)

    # =====================================================
    # START OPTIMIZATION
    # =====================================================
    if start_button:

        if "Terakhir" not in df.columns:

            st.error(
                "Kolom 'Terakhir' tidak ditemukan."
            )

        else:

            # =================================================
            # CLEAR SESSION
            # =================================================
            clear_session()
            gc.collect()

            # =================================================
            # STATUS
            # =================================================
            status_text = st.empty()

            progress_bar = st.progress(0)

            # =================================================
            # PREPROCESSING
            # =================================================
            status_text.write("⚙️ Preprocessing Data...")

            feature_cols = ["Terakhir"]
            target_col = "Terakhir"

            data_features = df[feature_cols].values
            data_target = df[[target_col]].values

            values = df[['Terakhir']].values

            n = len(values)

            n_train = int(n * 0.8)

            # =================================================
            # SCALING
            # =================================================
            scaler_X = MinMaxScaler().fit(
                data_features[:n_train]
            )

            scaler_y = MinMaxScaler().fit(
                data_target[:n_train]
            )

            Xs = scaler_X.transform(
                data_features
            )

            ys = scaler_y.transform(
                data_target
            )

            # =================================================
            # WINDOWING
            # =================================================
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

                return (
                    np.array(X_seq),
                    np.array(y_seq)
                )

            X_seq_all, y_seq_all = make_sequences(
                Xs,
                ys,
                timestep
            )

            dtrain_end = n_train - timestep

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

            train_size = int(
                len(X_train) * (1 - val_size)
            )

            X_tr = X_train[:train_size]
            y_tr = y_train[:train_size]

            X_val = X_train[train_size:]
            y_val = y_train[train_size:]

            # =================================================
            # PSO CONFIG
            # =================================================
            options = {
                'c1': 2.0,
                'c2': 2.0,
                'w': 0.7
            }

            bounds = (
                np.array([
                    units_min,
                    lr_min,
                    batch_min,
                    dropout_min
                ]),

                np.array([
                    units_max,
                    lr_max,
                    batch_max,
                    dropout_max
                ])
            )

            # =================================================
            # OBJECTIVE FUNCTION
            # =================================================
            def objective_function(particles_array):

                n_particles = particles_array.shape[0]

                losses = np.zeros(n_particles)

                for i, particle_i in enumerate(
                    particles_array
                ):

                    try:

                        units = int(
                            np.round(
                                particle_i[0]
                            )
                        )

                        lr = float(
                            particle_i[1]
                        )

                        batch = int(
                            np.round(
                                particle_i[2]
                            )
                        )

                        dropout = float(
                            particle_i[3]
                        )

                        units = max(1, units)

                        batch = max(1, batch)

                        clear_session()

                        tf.random.set_seed(SEED)

                        # =============================
                        # MODEL
                        # =============================
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

                        losses[i] = mse

                        clear_session()
                        gc.collect()

                    except Exception as e:

                        print("ERROR :", e)

                        losses[i] = 1e12

                return losses

            # =================================================
            # PSO OPTIMIZER
            # =================================================
            status_text.write(
                "🚀 Menjalankan PSO..."
            )

            optimizer = GlobalBestPSO(
                n_particles=particles,
                dimensions=4,
                options=options,
                bounds=bounds
            )

            best_cost, best_pos = optimizer.optimize(
                objective_function,
                iters=iterasi,
                verbose=True
            )

            progress_bar.progress(50)

            # =================================================
            # BEST PARAMETER
            # =================================================
            best_units = int(
                np.round(best_pos[0])
            )

            best_lr = float(
                best_pos[1]
            )

            best_batch = int(
                np.round(best_pos[2])
            )

            best_dropout = float(
                best_pos[3]
            )

            # =================================================
            # FINAL MODEL
            # =================================================
            status_text.write(
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

            progress_bar.progress(100)

            # =================================================
            # PREDICTION
            # =================================================
            status_text.write(
                "📊 Evaluasi Model..."
            )

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
            st.subheader(
                "🏆 Best Hyperparameter"
            )

            best_df = pd.DataFrame({

                "Units":
                [best_units],

                "Learning Rate":
                [best_lr],

                "Batch Size":
                [best_batch],

                "Dropout":
                [best_dropout]

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
            st.subheader(
                "📉 Grafik Konvergensi"
            )

            fig2, ax2 = plt.subplots(
                figsize=(10, 5)
            )

            ax2.plot(
                optimizer.cost_history,
                marker='o'
            )

            ax2.set_xlabel("Iterasi")

            ax2.set_ylabel("Best MSE")

            ax2.grid(alpha=0.3)

            st.pyplot(fig2)

            # =================================================
            # LOSS
            # =================================================
            st.subheader(
                "📉 Training Loss"
            )

            fig3, ax3 = plt.subplots(
                figsize=(10, 5)
            )

            ax3.plot(
                history.history['loss'],
                label='Training Loss'
            )

            ax3.plot(
                history.history['val_loss'],
                label='Validation Loss'
            )

            ax3.legend()

            ax3.grid(alpha=0.3)

            st.pyplot(fig3)

            # =================================================
            # AKTUAL VS PREDIKSI
            # =================================================
            st.subheader(
                "📈 Aktual vs Prediksi"
            )

            fig4, ax4 = plt.subplots(
                figsize=(12, 6)
            )

            ax4.plot(
                y_actual,
                label='Aktual'
            )

            ax4.plot(
                y_pred,
                label='Prediksi'
            )

            ax4.legend()

            ax4.grid(alpha=0.3)

            st.pyplot(fig4)

            st.success(
                "✅ Optimasi GRU-PSO selesai!"
            )

else:

    st.info(
        "📂 Silakan upload dataset terlebih dahulu."
    )
