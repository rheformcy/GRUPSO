import os

# =========================================================
# DETERMINISTIC
# =========================================================
os.environ["PYTHONHASHSEED"] = "49"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# =========================================================
# LIBRARY
# =========================================================
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
Aplikasi optimasi hyperparameter menggunakan
**
