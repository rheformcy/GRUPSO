import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Optimasi GRU-PSO",
    page_icon="🚀",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
# 🚀 Optimasi Gated Recurrent Unit - Particle Swarm Optimization (GRU-PSO)

### Rhena Amelia Shafitry  
Statistika Universitas Diponegoro  
24050122120019
""")

st.divider()

# ======================================================
# SIDEBAR - UPLOAD
# ======================================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

st.sidebar.caption(
    "Format file harus .xlsx atau .xls"
)

# ======================================================
# SIDEBAR - PARAMETER MODEL
# ======================================================
st.sidebar.header("⚙️ Parameter GRU")

timestep = st.sidebar.number_input(
    "Timestep / Window Size",
    min_value=1,
    max_value=30,
    value=1
)

layer = st.sidebar.number_input(
    "Jumlah Layer GRU",
    min_value=1,
    max_value=3,
    value=1
)

epoch = st.sidebar.number_input(
    "Epoch Final Training",
    min_value=1,
    value=50
)

# ======================================================
# SIDEBAR - PARAMETER PSO
# ======================================================
st.sidebar.header("🚀 Parameter PSO")

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

# ======================================================
# RANGE UNITS
# ======================================================
st.sidebar.subheader("Range Units")

units_min = st.sidebar.number_input(
    "Units Minimum",
    min_value=1,
    value=16
)

units_max = st.sidebar.number_input(
    "Units Maximum",
    min_value=1,
    value=128
)

# ======================================================
# RANGE LEARNING RATE
# ======================================================
st.sidebar.subheader("Range Learning Rate")

lr_min = st.sidebar.number_input(
    "Learning Rate Minimum",
    min_value=0.0001,
    value=0.0001,
    step=0.0001,
    format="%.4f"
)

lr_max = st.sidebar.number_input(
    "Learning Rate Maximum",
    min_value=0.0001,
    value=0.01,
    step=0.0001,
    format="%.4f"
)

# ======================================================
# RANGE BATCH SIZE
# ======================================================
st.sidebar.subheader("Range Batch Size")

batch_min = st.sidebar.number_input(
    "Batch Size Minimum",
    min_value=1,
    value=16
)

batch_max = st.sidebar.number_input(
    "Batch Size Maximum",
    min_value=1,
    value=128
)

# ======================================================
# RANGE DROPOUT
# ======================================================
st.sidebar.subheader("Range Dropout")

dropout_min = st.sidebar.slider(
    "Dropout Minimum",
    min_value=0.0,
    max_value=0.9,
    value=0.1,
    step=0.1
)

dropout_max = st.sidebar.slider(
    "Dropout Maximum",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.1
)

# ======================================================
# MAIN PAGE
# ======================================================
if uploaded_file is not None:

    try:

        # ==================================================
        # LOAD DATA
        # ==================================================
        df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip()

        # ==================================================
        # TABS
        # ==================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Preview Dataset",
            "📊 Statistik Data",
            "📈 Visualisasi",
            "🚀 Optimasi GRU-PSO"
        ])

        # ==================================================
        # TAB 1 - PREVIEW
        # ==================================================
        with tab1:

            st.subheader("📄 Preview Dataset")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            st.write(f"Jumlah Baris : {df.shape[0]}")
            st.write(f"Jumlah Kolom : {df.shape[1]}")

        # ==================================================
        # TAB 2 - DESKRIPTIF
        # ==================================================
        with tab2:

            st.subheader("📊 Statistik Deskriptif")

            numeric_df = df.select_dtypes(
                include=['int64', 'float64']
            )

            st.dataframe(
                numeric_df.describe(),
                use_container_width=True
            )

        # ==================================================
        # TAB 3 - VISUALISASI
        # ==================================================
        with tab3:

            st.subheader("📈 Visualisasi Time Series")

            if (
                "Tanggal" in df.columns and
                "Terakhir" in df.columns
            ):

                st.line_chart(
                    data=df,
                    x="Tanggal",
                    y="Terakhir",
                    use_container_width=True
                )

            else:

                st.warning(
                    "Kolom 'Tanggal' dan "
                    "'Terakhir' tidak ditemukan."
                )

        # ==================================================
        # TAB 4 - GRU PSO
        # ==================================================
        with tab4:

            st.subheader("🚀 Optimasi GRU-PSO")

            # ==============================================
            # INFO PARAMETER
            # ==============================================
            st.markdown("### ⚙️ Parameter Model")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Timestep",
                timestep
            )

            col2.metric(
                "Layer",
                layer
            )

            col3.metric(
                "Epoch",
                epoch
            )

            st.divider()

            st.markdown("### 🚀 Parameter PSO")

            col4, col5 = st.columns(2)

            col4.metric(
                "Particle",
                particle
            )

            col5.metric(
                "Iterasi",
                iterasi
            )

            st.divider()

            # ==============================================
            # BUTTON MULAI OPTIMASI
            # ==============================================
            mulai = st.button(
                "🚀 Mulai Optimasi",
                use_container_width=True,
                type="primary"
            )

            # ==============================================
            # JIKA BUTTON DIKLIK
            # ==============================================
            if mulai:

                st.info(
                    "Model GRU-PSO sedang dijalankan..."
                )

                # ==========================================
                # PROGRESS BAR
                # ==========================================
                progress_bar = st.progress(0)

                status_text = st.empty()

                for i in range(100):

                    progress_bar.progress(i + 1)

                    status_text.text(
                        f"Progress Optimasi: {i+1}%"
                    )

                st.success(
                    "Optimasi selesai!"
                )

                # ==========================================
                # PLACEHOLDER HASIL
                # ==========================================
                st.subheader("🏆 Best Hyperparameter")

                dummy_result = pd.DataFrame({

                    "Units": [64],
                    "Learning Rate": [0.0012],
                    "Batch Size": [32],
                    "Dropout": [0.2]

                })

                st.dataframe(
                    dummy_result,
                    use_container_width=True
                )

                # ==========================================
                # METRICS PLACEHOLDER
                # ==========================================
                st.subheader("📊 Evaluasi Model")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "RMSE",
                    "12,345.67"
                )

                m2.metric(
                    "MAE",
                    "10,876.54"
                )

                m3.metric(
                    "MAPE",
                    "1.2345%"
                )

    except Exception as e:

        st.error(
            f"❌ Terjadi error: {e}"
        )

else:

    st.info(
        "📂 Silakan upload file Excel terlebih dahulu."
    )
