import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL DAN ENCODER
# ==============================
model = joblib.load('model_naive_bayes.pkl')
le_refund = joblib.load('le_refund.pkl')
le_marital = joblib.load('le_marital.pkl')
le_evasion = joblib.load('le_evasion.pkl')

# ==============================
# JUDUL APLIKASI
# ==============================
st.set_page_config(page_title='Prediksi Tax Evasion', layout='centered')
st.title('Prediksi Tax Evasion Menggunakan Naive Bayes')
st.write('Aplikasi ini digunakan untuk memprediksi kemungkinan **Tax Evasion** berdasarkan data wajib pajak.')

# ==============================
# INPUT USER
# ==============================
st.header('Input Data Wajib Pajak')

refund = st.selectbox('Refund', ['Yes', 'No'])
marital = st.selectbox('Status Pernikahan', ['Single', 'Married', 'Divorced'])
income = st.number_input('Income', min_value=0, step=1000)

# ==============================
# TOMBOL PREDIKSI
# ==============================
if st.button('Prediksi'):
    # Encoding input
    refund_enc = le_refund.transform([refund])[0]
    marital_enc = le_marital.transform([marital])[0]

    # DataFrame input
    data_baru = pd.DataFrame([[refund_enc, marital_enc, income]],
                             columns=['Refund', 'Marital', 'Income'])

    # Prediksi
    prediksi = model.predict(data_baru)
    probabilitas = model.predict_proba(data_baru)

    hasil = le_evasion.inverse_transform(prediksi)[0]

    # ==============================
    # OUTPUT
    # ==============================
    st.subheader('Hasil Prediksi')
    st.write(f'**Prediksi Kelas:** {hasil}')

    st.subheader('Probabilitas')
    prob_df = pd.DataFrame(probabilitas, columns=le_evasion.classes_)
    st.dataframe(prob_df)

    if probabilitas[0][0] > probabilitas[0][1]:
        st.success('Kesimpulan: Wajib pajak **TIDAK melakukan tax evasion**')
    else:
        st.error('Kesimpulan: Wajib pajak **BERPOTENSI melakukan tax evasion**')

# ==============================
# CATATAN
# ==============================
st.markdown('---')
st.caption('Model: Gaussian Naive Bayes | Dataset kecil (contoh akademik)')
