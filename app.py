import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="GRU-PSO Forecasting",
    layout="wide"
)

st.title("📈 Prediksi Harga Emas GRU-PSO")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_gru_model():

    model = load_model(
        "gru_pso_model.keras"
    )

    scaler = joblib.load(
        "scaler.save"
    )

    return model, scaler

model, scaler = load_gru_model()

# =====================================================
# UPLOAD FILE
# =====================================================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset Excel",
    type=["xlsx"]
)

if uploaded_file is not None:

    # =================================================
    # READ DATA
    # =================================================
    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    df["Terakhir"] = pd.to_numeric(
        df["Terakhir"],
        errors="coerce"
    )

    df = df.dropna().reset_index(drop=True)

    st.subheader("📋 Dataset")

    st.dataframe(df.head())

    # =================================================
    # VISUALISASI
    # =================================================
    st.subheader("📈 Visualisasi Data")

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(df["Terakhir"])

    ax.grid(True)

    st.pyplot(fig)

    # =================================================
    # SCALING
    # =================================================
    values = df[["Terakhir"]].values

    scaled_all = scaler.transform(values)

    # =================================================
    # WINDOWING
    # =================================================
    WINDOW = 5

    def make_sequences(data, window):

        X = []
        y = []

        for i in range(window, len(data)):

            X.append(data[i-window:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    X_all, y_all = make_sequences(
        scaled_all,
        WINDOW
    )

    # =================================================
    # SPLIT
    # =================================================
    n_train = int(len(values) * 0.8)

    train_end = n_train - WINDOW

    X_test = X_all[train_end:]
    y_test = y_all[train_end:]

    X_test = X_test.reshape(
        (
            X_test.shape[0],
            X_test.shape[1],
            1
        )
    )

    # =================================================
    # PREDICT
    # =================================================
    y_pred_scaled = model.predict(
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

    st.subheader("📊 Evaluasi")

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
    # PLOT
    # =================================================
    st.subheader("📈 Actual vs Prediction")

    fig2, ax2 = plt.subplots(
        figsize=(14, 7)
    )

    ax2.plot(
        y_actual,
        label="Actual",
        linewidth=2
    )

    ax2.plot(
        y_pred,
        label="Prediction",
        linestyle="--",
        linewidth=2
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(fig2)
