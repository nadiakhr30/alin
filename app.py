import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title='Prediksi Tax Evasion',
    page_icon='💼',
    layout='centered'
)

# ==============================
# LOAD MODEL & ENCODER
# ==============================
model = joblib.load('model_naive_bayes.pkl')
le_refund = joblib.load('le_refund.pkl')
le_marital = joblib.load('le_marital.pkl')
le_evasion = joblib.load('le_evasion.pkl')

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📌 Informasi Model")
st.sidebar.markdown("**Algoritma:** Gaussian Naive Bayes")
st.sidebar.markdown("**Jumlah Kelas:** 2 (No, Yes)")
st.sidebar.metric("Akurasi Model (Test Data)", "66.67%")
st.sidebar.caption(
    "Akurasi diperoleh dari evaluasi model "
    "menggunakan data uji (testing set)."
)

# ==============================
# HERO SECTION
# ==============================
st.markdown("""
<div style="background: linear-gradient(90deg,#1f4037,#99f2c8);
            padding:30px;border-radius:15px">
    <h1 style="color:white;text-align:center;">💼 Prediksi Tax Evasion</h1>
    <p style="color:white;text-align:center;font-size:16px">
        Sistem Pendukung Keputusan Berbasis Machine Learning<br>
        <b>Algoritma Gaussian Naive Bayes</b>
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ==============================
# DESKRIPSI
# ==============================
with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    Aplikasi ini digunakan untuk memprediksi **kemungkinan terjadinya Tax Evasion**
    berdasarkan:
    - Refund
    - Status Pernikahan
    - Income
    
    Model menggunakan **Gaussian Naive Bayes**
    dan ditujukan untuk **keperluan akademik**.
    """)

# ==============================
# INPUT DATA
# ==============================
st.subheader("📝 Input Data Wajib Pajak")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        refund = st.selectbox("Refund", ["Yes", "No"])
        income = st.number_input("Income (Rp)", min_value=0, step=1000)

    with col2:
        marital = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])

    submit = st.form_submit_button("🔍 Proses Prediksi", use_container_width=True)

# ==============================
# PROSES PREDIKSI
# ==============================
if submit:
    if income == 0:
        st.warning("⚠️ Income tidak boleh 0")
    else:
        refund_enc = le_refund.transform([refund])[0]
        marital_enc = le_marital.transform([marital])[0]

        data_input = pd.DataFrame(
            [[refund_enc, marital_enc, income]],
            columns=["Refund", "Marital", "Income"]
        )

        prediksi = model.predict(data_input)
        probabilitas = model.predict_proba(data_input)
        hasil = le_evasion.inverse_transform(prediksi)[0]

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if hasil == "No":
            st.success("✅ **KESIMPULAN:** TIDAK melakukan Tax Evasion")
        else:
            st.error("⚠️ **KESIMPULAN:** BERPOTENSI melakukan Tax Evasion")

        # ==============================
        # TABEL PROBABILITAS
        # ==============================
        st.markdown("### 📈 Probabilitas Kelas")
        prob_df = pd.DataFrame(
            probabilitas,
            columns=le_evasion.classes_
        )
        st.dataframe(prob_df, use_container_width=True)

        # ==============================
        # BAR CHART
        # ==============================
        st.markdown("### 📊 Bar Chart Probabilitas")
        chart_df = prob_df.T
        chart_df.columns = ["Probabilitas"]
        st.bar_chart(chart_df)

    
        # ==============================
        # CONFIDENCE BAR
        # ==============================
        confidence = max(probabilitas[0]) * 100
        st.progress(int(confidence))
        st.caption(f"Tingkat keyakinan prediksi: **{confidence:.2f}%**")

        st.info(
            "Nilai keyakinan berasal dari probabilitas posterior "
            "untuk satu data input, bukan akurasi model secara keseluruhan."
        )

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("© 2025 | Prediksi Tax Evasion | Gaussian Naive Bayes | Akademik")
