import streamlit as st
import pandas as pd
import joblib

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title='Prediksi Tax Evasion',
    page_icon='💼',
    layout='centered'
)

# ==============================
# LOAD MODEL DAN ENCODER
# ==============================
model = joblib.load('model_naive_bayes.pkl')
le_refund = joblib.load('le_refund.pkl')
le_marital = joblib.load('le_marital.pkl')
le_evasion = joblib.load('le_evasion.pkl')

# ==============================
# HEADER UTAMA (HERO SECTION)
# ==============================
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1f4037, #99f2c8); 
                padding: 25px; border-radius: 12px;">
        <h1 style="color:white; text-align:center;">💼 Prediksi Tax Evasion</h1>
        <p style="color:white; text-align:center; font-size:16px;">
            Sistem pendukung keputusan berbasis <b>Machine Learning</b><br>
            Menggunakan algoritma <b>Gaussian Naive Bayes</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ==============================
# DESKRIPSI APLIKASI
# ==============================
with st.expander("ℹ️ Tentang Aplikasi"):
    st.write(
        "Aplikasi ini digunakan untuk memprediksi kemungkinan terjadinya **Tax Evasion** "
        "berdasarkan atribut **Refund**, **Status Pernikahan**, dan **Income**. "
        "Model yang digunakan adalah **Gaussian Naive Bayes** dan ditujukan untuk "
        "keperluan akademik dan pembelajaran."
    )

# ==============================
# FORM INPUT
# ==============================
st.subheader('📝 Input Data Wajib Pajak')

with st.form(key='form_prediksi'):
    col1, col2 = st.columns(2)

    with col1:
        refund = st.selectbox('Refund', ['Yes', 'No'])
        income = st.number_input('Income (Rp)', min_value=0, step=1000)

    with col2:
        marital = st.selectbox('Status Pernikahan', ['Single', 'Married', 'Divorced'])

    submit = st.form_submit_button('🔍 Proses Prediksi', use_container_width=True)

# ==============================
# PROSES PREDIKSI
# ==============================
if submit:
    refund_enc = le_refund.transform([refund])[0]
    marital_enc = le_marital.transform([marital])[0]

    data_baru = pd.DataFrame(
        [[refund_enc, marital_enc, income]],
        columns=['Refund', 'Marital', 'Income']
    )

    prediksi = model.predict(data_baru)
    probabilitas = model.predict_proba(data_baru)
    hasil = le_evasion.inverse_transform(prediksi)[0]

    st.markdown('---')
    st.subheader('📊 Hasil Analisis')

    if hasil == 'No':
        st.success('✅ **KESIMPULAN:** Wajib pajak **TIDAK melakukan Tax Evasion**')
    else:
        st.error('⚠️ **KESIMPULAN:** Wajib pajak **BERPOTENSI melakukan Tax Evasion**')

    st.markdown('### 📈 Probabilitas Kelas')
    prob_df = pd.DataFrame(probabilitas, columns=le_evasion.classes_)
    st.dataframe(prob_df, use_container_width=True)

    st.progress(int(max(probabilitas[0]) * 100))
    st.caption(f"Tingkat keyakinan model: {max(probabilitas[0]) * 100:.2f}%")

    st.info(
        "Keputusan ditentukan berdasarkan nilai probabilitas posterior terbesar "
        "sesuai dengan prinsip algoritma Naive Bayes."
    )

# ==============================
# FOOTER
# ==============================
st.markdown('---')
st.caption('© 2025 | Aplikasi Prediksi Tax Evasion | Gaussian Naive Bayes | Akademik')