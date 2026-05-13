import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
import tensorflow as tf
import pyswarms as ps

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

# =========================================================
# CONFIG
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
Aplikasi optimasi hyperparameter  
menggunakan **Particle Swarm Optimization (PSO)**  
pada model **Gated Recurrent Unit (GRU)**.
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
st.sidebar.header("⚙️ Parameter Model")

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
    0.0,
    0.9,
    0.1
)

dropout_max = st.sidebar.slider(
    "Dropout Max",
    0.0,
    0.9,
    0.5
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
    st.subheader("🚨 Tabel Outlier")

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
    # VISUALISASI
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
            # SEED
            # =================================================
            SEED = 49

            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            # =================================================
            # CLEAR
            # =================================================
            clear_session()
            gc.collect()

            # =================================================
            # STATUS
            # =================================================
            status = st.empty()

            progress_bar = st.progress(0)

            # =================================================
            # PREPROCESSING
            # =================================================
            status.write("⚙️ Preprocessing data...")

            values = df[['Terakhir']].values.astype(float)

            n = len(values)

            n_train = int(n * 0.8)

            train_values = values[:n_train]

            scaler = MinMaxScaler()

            scaler.fit(train_values)

            scaled_data = scaler.transform(values)

            # =================================================
            # WINDOWING
            # =================================================
            X = []
            y = []

            for i in range(timestep, len(scaled_data)):

                X.append(
                    scaled_data[i-timestep:i]
                )

                y.append(
                    scaled_data[i]
                )

            X = np.array(X)
            y = np.array(y)

            split_idx = n_train - timestep

            X_train = X[:split_idx]
            y_train = y[:split_idx]

            X_test = X[split_idx:]
            y_test = y[split_idx:]

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
            # FITNESS FUNCTION
            # =================================================
            def objective_function(particles_array):

                n_particles = particles_array.shape[0]

                losses = np.zeros(n_particles)

                for i, particle_i in enumerate(particles_array):

                    try:

                        units = int(
                            np.round(particle_i[0])
                        )

                        lr = float(
                            particle_i[1]
                        )

                        batch = int(
                            np.round(particle_i[2])
                        )

                        dropout = float(
                            particle_i[3]
                        )

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
                            epochs=epochs_pso,
                            batch_size=batch,
                            verbose=0
                        )

                        pred = model.predict(
                            X_val,
                            verbose=0
                        )

                        pred_inv = scaler.inverse_transform(
                            pred
                        ).flatten()

                        actual_inv = scaler.inverse_transform(
                            y_val.reshape(-1, 1)
                        ).flatten()

                        mse = mean_squared_error(
                            actual_inv,
                            pred_inv
                        )

                        losses[i] = mse

                        clear_session()
                        gc.collect()

                    except:

                        losses[i] = 1e12

                return losses

            # =================================================
            # OPTIMIZER
            # =================================================
            status.write("🚀 Menjalankan optimasi PSO...")

            optimizer = ps.single.GlobalBestPSO(
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
            best_units = int(np.round(best_pos[0]))
            best_lr = float(best_pos[1])
            best_batch = int(np.round(best_pos[2]))
            best_dropout = float(best_pos[3])

            # =================================================
            # FINAL TRAINING
            # =================================================
            status.write("🤖 Training final model...")

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
                verbose=1
            )

            progress_bar.progress(100)

            # =================================================
            # PREDICTION
            # =================================================
            status.write("📊 Evaluasi model...")

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
            st.subheader("📉 Grafik Konvergensi")

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
            st.subheader("📉 Training Loss")

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
            st.subheader("📈 Aktual vs Prediksi")

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
