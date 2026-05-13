import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import gc
import random

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# SKLEARN
# ======================================================
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# ======================================================
# TENSORFLOW
# ======================================================
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

# ======================================================
# PSO
# ======================================================
import pyswarms as ps

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="GRU-PSO Gold Forecasting",
    layout="wide"
)

# ======================================================
# TITLE
# ======================================================
st.title("📈 Optimasi GRU-PSO Harga Emas")

st.markdown("""
Aplikasi Optimasi Hyperparameter  
Menggunakan:
- GRU (Gated Recurrent Unit)
- Particle Swarm Optimization (PSO)
""")

st.divider()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

st.sidebar.divider()

# ======================================================
# PARAMETER
# ======================================================
st.sidebar.header("⚙️ Parameter")

timestep = st.sidebar.number_input(
    "Timestep",
    min_value=1,
    max_value=30,
    value=1
)

particle = st.sidebar.number_input(
    "Jumlah Partikel",
    min_value=1,
    value=40
)

iterasi = st.sidebar.number_input(
    "Jumlah Iterasi",
    min_value=1,
    value=10
)

epoch_final = st.sidebar.number_input(
    "Epoch Final",
    min_value=1,
    value=50
)

# ======================================================
# BUTTON
# ======================================================
mulai_optimasi = st.button(
    "🚀 Mulai Optimasi",
    use_container_width=True
)

# ======================================================
# MAIN
# ======================================================
if uploaded_file is not None:

    try:

        # ==================================================
        # LOAD DATA
        # ==================================================
        df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip()

        # ==================================================
        # CLEANING
        # ==================================================
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

        df = df.dropna()

        # ==================================================
        # VALIDASI KOLOM
        # ==================================================
        if "Terakhir" not in df.columns:

            st.error(
                "Kolom 'Terakhir' tidak ditemukan."
            )

            st.stop()

        # ==================================================
        # NUMERIC
        # ==================================================
        df["Terakhir"] = pd.to_numeric(
            df["Terakhir"],
            errors='coerce'
        )

        df = df.dropna()

        # ==================================================
        # PREVIEW
        # ==================================================
        st.subheader("📄 Preview Dataset")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # ==================================================
        # MULAI OPTIMASI
        # ==================================================
        if mulai_optimasi:

            # ==============================================
            # CLEAR
            # ==============================================
            clear_session()

            gc.collect()

            # ==============================================
            # SET SEED
            # ==============================================
            SEED = 49

            random.seed(SEED)

            np.random.seed(SEED)

            tf.random.set_seed(SEED)

            tf.keras.utils.set_random_seed(SEED)

            tf.config.experimental.enable_op_determinism()

            # ==============================================
            # PREPROCESSING
            # ==============================================
            with st.spinner(
                "Melakukan preprocessing data..."
            ):

                feature_cols = ["Terakhir"]

                target_col = "Terakhir"

                data_features = df[
                    feature_cols
                ].values

                data_target = df[
                    [target_col]
                ].values

                # ==========================================
                # SPLIT
                # ==========================================
                values = df[
                    ['Terakhir']
                ].values

                n = len(values)

                n_train = int(n * 0.8)

                # ==========================================
                # SCALER
                # ==========================================
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

                # ==========================================
                # WINDOWING
                # ==========================================
                def make_sequences(
                    X_scaled,
                    y_scaled,
                    window
                ):

                    X_seq = []

                    y_seq = []

                    for i in range(
                        window,
                        len(X_scaled)
                    ):

                        X_seq.append(
                            X_scaled[
                                i-window:i
                            ]
                        )

                        y_seq.append(
                            y_scaled[i]
                        )

                    return (
                        np.array(X_seq),
                        np.array(y_seq)
                    )

                X_seq_all, y_seq_all = (
                    make_sequences(
                        Xs,
                        ys,
                        window=timestep
                    )
                )

                dtrain_end = (
                    n_train - timestep
                )

                X_train = X_seq_all[
                    :dtrain_end
                ]

                y_train = y_seq_all[
                    :dtrain_end
                ]

                X_test = X_seq_all[
                    dtrain_end:
                ]

                y_test = y_seq_all[
                    dtrain_end:
                ]

                # ==========================================
                # RESHAPE
                # ==========================================
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

                st.success(
                    "Preprocessing selesai!"
                )

            # ==============================================
            # VALIDATION
            # ==============================================
            val_size = 0.2

            n_tr_samples = X_train.shape[0]

            n_tr_val = int(
                n_tr_samples * (1 - val_size)
            )

            X_tr = X_train[:n_tr_val]

            y_tr = y_train[:n_tr_val]

            X_val = X_train[n_tr_val:]

            y_val = y_train[n_tr_val:]

            # ==============================================
            # PSO CONFIG
            # ==============================================
            PSO_options = {
                'c1': 2.0,
                'c2': 2.0,
                'w': 0.7
            }

            PSO_bounds = (
                [16, 0.0001, 16, 0.1],
                [128, 0.01, 128, 0.5]
            )

            # ==============================================
            # OBJECTIVE FUNCTION
            # ==============================================
            def make_pso_obj(
                X_tr,
                y_tr,
                X_va,
                y_va,
                scaler_y
            ):

                def obj_fn(particles):

                    n_particles = (
                        particles.shape[0]
                    )

                    costs = np.zeros(
                        n_particles
                    )

                    for i, p in enumerate(
                        particles
                    ):

                        units = int(
                            np.round(p[0])
                        )

                        lr = float(p[1])

                        batch = int(
                            np.round(p[2])
                        )

                        dropout = float(
                            p[3]
                        )

                        try:

                            clear_session()

                            tf.keras.utils.set_random_seed(
                                SEED
                            )

                            # ======================
                            # MODEL
                            # ======================
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

                                Dropout(
                                    dropout
                                ),

                                Dense(1)

                            ])

                            model.compile(
                                optimizer=Adam(
                                    learning_rate=lr
                                ),
                                loss='mse'
                            )

                            # ======================
                            # TRAIN
                            # ======================
                            model.fit(
                                X_tr,
                                y_tr,
                                epochs=10,
                                batch_size=batch,
                                verbose=0,
                                shuffle=False
                            )

                            # ======================
                            # PREDICT
                            # ======================
                            yv_pred = model.predict(
                                X_va,
                                verbose=0
                            )

                            yv_pred_orig = (
                                scaler_y.inverse_transform(
                                    yv_pred
                                ).flatten()
                            )

                            yv_true_orig = (
                                scaler_y.inverse_transform(
                                    y_va.reshape(-1, 1)
                                ).flatten()
                            )

                            costs[i] = (
                                mean_squared_error(
                                    yv_true_orig,
                                    yv_pred_orig
                                )
                            )

                        except Exception as e:

                            print(
                                "PSO ERROR:",
                                e
                            )

                            costs[i] = 1e12

                        clear_session()

                        gc.collect()

                    return costs

                return obj_fn

            # ==============================================
            # PSO OBJECTIVE
            # ==============================================
            pso_obj = make_pso_obj(
                X_tr,
                y_tr,
                X_val,
                y_val,
                scaler_y
            )

            # ==============================================
            # OPTIMIZER
            # ==============================================
            optimizer = ps.single.GlobalBestPSO(
                n_particles=particle,
                dimensions=4,
                options=PSO_options,
                bounds=PSO_bounds
            )

            # ==============================================
            # HISTORY
            # ==============================================
            history_gbest = []

            # ==============================================
            # PROGRESS BAR
            # ==============================================
            progress_bar = st.progress(0)

            status_text = st.empty()

            # ==============================================
            # MANUAL LOOP PSO
            # ==============================================
            for it in range(iterasi):

                status_text.write(
                    f"🚀 Iterasi PSO ke-{it+1}/{iterasi}"
                )

                costs = pso_obj(
                    optimizer.swarm.position
                )

                # ==========================================
                # UPDATE PBEST
                # ==========================================
                mask = (
                    costs <
                    optimizer.swarm.pbest_cost
                )

                optimizer.swarm.pbest_cost[
                    mask
                ] = costs[mask]

                optimizer.swarm.pbest_pos[
                    mask
                ] = (
                    optimizer.swarm.position[
                        mask
                    ]
                )

                # ==========================================
                # UPDATE GBEST
                # ==========================================
                best_idx = np.argmin(
                    optimizer.swarm.pbest_cost
                )

                optimizer.swarm.best_cost = (
                    optimizer.swarm.pbest_cost[
                        best_idx
                    ]
                )

                optimizer.swarm.best_pos = (
                    optimizer.swarm.pbest_pos[
                        best_idx
                    ]
                )

                history_gbest.append(
                    optimizer.swarm.best_cost
                )

                # ==========================================
                # VELOCITY
                # ==========================================
                r1 = np.random.rand(
                    *optimizer.swarm.position.shape
                )

                r2 = np.random.rand(
                    *optimizer.swarm.position.shape
                )

                optimizer.swarm.velocity = (

                    PSO_options['w']
                    * optimizer.swarm.velocity

                    +

                    PSO_options['c1']
                    * r1
                    * (
                        optimizer.swarm.pbest_pos
                        - optimizer.swarm.position
                    )

                    +

                    PSO_options['c2']
                    * r2
                    * (
                        optimizer.swarm.best_pos
                        - optimizer.swarm.position
                    )
                )

                # ==========================================
                # POSITION
                # ==========================================
                optimizer.swarm.position += (
                    optimizer.swarm.velocity
                )

                lb = np.array(
                    PSO_bounds[0]
                )

                ub = np.array(
                    PSO_bounds[1]
                )

                optimizer.swarm.position = np.clip(
                    optimizer.swarm.position,
                    lb,
                    ub
                )

                # ==========================================
                # PROGRESS
                # ==========================================
                progress_bar.progress(
                    (it + 1) / iterasi
                )

            # ==============================================
            # BEST PARAMETER
            # ==============================================
            best_pos = (
                optimizer.swarm.best_pos
            )

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

            # ==============================================
            # FINAL MODEL
            # ==============================================
            with st.spinner(
                "Training final model..."
            ):

                clear_session()

                tf.keras.utils.set_random_seed(
                    SEED
                )

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

                    Dropout(
                        best_dropout
                    ),

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
                    epochs=epoch_final,
                    batch_size=best_batch,
                    validation_split=0.2,
                    verbose=1,
                    shuffle=False
                )

            # ==============================================
            # PREDICT
            # ==============================================
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

            # ==============================================
            # METRICS
            # ==============================================
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

            # ==============================================
            # BEST PARAMETER
            # ==============================================
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

            # ==============================================
            # METRICS
            # ==============================================
            st.subheader(
                "📊 Evaluasi Model"
            )

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

            # ==============================================
            # KONVERGENSI
            # ==============================================
            st.subheader(
                "📉 Grafik Konvergensi PSO"
            )

            fig1, ax1 = plt.subplots(
                figsize=(10, 5)
            )

            ax1.plot(
                history_gbest,
                marker='o'
            )

            ax1.set_xlabel(
                "Iterasi"
            )

            ax1.set_ylabel(
                "Best MSE"
            )

            ax1.grid(alpha=0.3)

            st.pyplot(fig1)

            # ==============================================
            # LOSS
            # ==============================================
            st.subheader(
                "📉 Training vs Validation Loss"
            )

            fig2, ax2 = plt.subplots(
                figsize=(10, 5)
            )

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

            # ==============================================
            # PREDIKSI
            # ==============================================
            st.subheader(
                "📈 Aktual vs Prediksi"
            )

            fig3, ax3 = plt.subplots(
                figsize=(12, 6)
            )

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

            # ==============================================
            # RESULT TABLE
            # ==============================================
            result_df = pd.DataFrame({

                "RMSE": [rmse],
                "MAE": [mae],
                "MAPE": [mape],
                "Units": [best_units],
                "Learning Rate": [best_lr],
                "Batch Size": [best_batch],
                "Dropout": [best_dropout]

            })

            st.subheader(
                "📋 Hasil Akhir"
            )

            st.dataframe(
                result_df,
                use_container_width=True
            )

            st.success(
                "Optimasi GRU-PSO selesai!"
            )

    except Exception as e:

        st.error(
            f"❌ Terjadi error: {e}"
        )

else:

    st.info(
        "📂 Silakan upload file Excel terlebih dahulu."
    )
