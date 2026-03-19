import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- 1. LOAD RESOURCES (MODEL & ENCODERS) ---
@st.cache_resource
def load_resources():
    # Load model & encoders dari folder models/
    model = joblib.load('models/random_forest_best.pkl')
    ohe_intent = joblib.load('models/ohe_loan_intent.pkl')
    ohe_grade = joblib.load('models/ohe_loan_grade.pkl')
    ohe_ownership = joblib.load('models/ohe_home_ownership.pkl')
    
    try:
        ohe_default = joblib.load('models/ohe_default_on_file.pkl')
    except:
        ohe_default = None
        
    # Pakai threshold manual 0.32 agar tidak error KeyError JSON
    threshold = 0.32
        
    return model, ohe_intent, ohe_grade, ohe_ownership, ohe_default, threshold

model, ohe_intent, ohe_grade, ohe_ownership, ohe_default, threshold = load_resources()

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Internal Credit Risk System", layout="wide")

col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.title("🏦") 
with col_text:
    st.title("Credit Risk Assessment Dashboard")
    st.write("**Internal Tools for Credit Officers & Analysts**")

st.image("https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
         caption="Sistem Prediksi Probabilitas Gagal Bayar (Default Probability)", use_container_width=True)

# --- 3. PANDUAN PENGGUNA ---
with st.expander("📖 Panduan Operasional & Referensi Parameter (KLIK DI SINI)"):
    st.markdown("""
    ### **Logika Bisnis & Threshold:**
    * **Threshold:** 0.32 (Prediksi di atas 32% dianggap risiko tinggi/gagal bayar).
    * **Loan Grade:** Menentukan tingkat risiko awal nasabah.
    """)

st.divider()

# --- 4. INPUT USER ---
st.subheader("📝 Data Pengajuan Kredit")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### **🔵 Identitas & Kemampuan Nasabah**")
    person_age = st.number_input("Usia Nasabah (Tahun):", 18, 80, 25)
    person_income = st.number_input("Pendapatan Tahunan (USD):", 1000, 1000000, 50000)
    person_emp_length = st.number_input("Lama Bekerja (Tahun):", 0.0, 50.0, 2.0)
    person_home_ownership = st.selectbox("Status Kepemilikan Rumah:", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (Tahun):", 0, 40, 3)

with col2:
    st.markdown("### **🔴 Parameter & Kebijakan Internal Bank**")
    loan_intent = st.selectbox("Tujuan Pinjaman:", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade (Hasil Scoring Awal):", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Jumlah Pinjaman yang Diajukan (USD):", 500, 50000, 5000)
    loan_int_rate = st.number_input("Suku Bunga Penawaran (%):", 0.1, 25.0, 11.0)
    cb_person_default_on_file = st.selectbox("Pernah Gagal Bayar Sebelumnya (Data SLIK)?", ["N", "Y"])

loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
st.metric("Debt to Income Ratio (DTI)", f"{loan_percent_income}")

# --- 5. PROSES PREDIKSI ---
if st.button("🚀 Jalankan Prediksi"):
    if person_age > 65:
        st.error("❌ **AKSES DITOLAK:** Nasabah berada di luar usia produktif.")
    elif person_income < 5000:
        st.error("❌ **AKSES DITOLAK:** Pendapatan tidak memenuhi batas minimum.")
    else:
        with st.spinner("🤖 AI sedang menganalisis..."):
            try:
                # 1. Buat DataFrame Input
                input_df = pd.DataFrame([{
                    'person_age': person_age,
                    'person_income': person_income,
                    'person_emp_length': person_emp_length,
                    'loan_amnt': loan_amnt,
                    'loan_int_rate': loan_int_rate,
                    'loan_percent_income': loan_percent_income,
                    'cb_person_cred_hist_length': cb_person_cred_hist_length,
                    'loan_intent': loan_intent,
                    'loan_grade': loan_grade,
                    'person_home_ownership': person_home_ownership,
                    'cb_person_default_on_file': cb_person_default_on_file
                }])

                # 2. Transform Categorical (One-Hot Encoding)
                # Gunakan Prefix yang sama dengan modeling agar nama kolom cocok
                owner_f = pd.DataFrame(ohe_ownership.transform(input_df[['person_home_ownership']]).toarray(), 
                                       columns=ohe_ownership.get_feature_names_out(['person_home_ownership']))
                intent_f = pd.DataFrame(ohe_intent.transform(input_df[['loan_intent']]).toarray(), 
                                        columns=ohe_intent.get_feature_names_out(['loan_intent']))
                grade_f = pd.DataFrame(ohe_grade.transform(input_df[['loan_grade']]).toarray(), 
                                       columns=ohe_grade.get_feature_names_out(['loan_grade']))
                default_f = pd.DataFrame(ohe_default.transform(input_df[['cb_person_default_on_file']]).toarray(), 
                                         columns=ohe_default.get_feature_names_out(['cb_person_default_on_file']))

                # 3. Gabung dengan urutan kolom yang benar (Sesuai modeling Ety)
                # Sesuaikan nama kolom ohe agar prefix-nya pas (misal: home_ownership_RENT)
                # Kita ganti nama kolom manual agar sesuai dengan 'column_order' di API
                owner_f.columns = [c.replace('person_home_ownership', 'home_ownership') for c in owner_f.columns]
                intent_f.columns = [c.replace('loan_intent', 'loan_intent') for c in intent_f.columns]
                grade_f.columns = [c.replace('loan_grade', 'loan_grade') for c in grade_f.columns]
                default_f.columns = [c.replace('cb_person_default_on_file', 'default_onfile') for c in default_f.columns]

                final_df = pd.concat([
                    input_df[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                              'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']],
                    owner_f, intent_f, grade_f, default_f
                ], axis=1)

                # 4. Predict
                prob = model.predict_proba(final_df)[:, 1][0]
                proba_persen = prob * 100
                st.divider()

                if prob >= threshold:
                    st.error(f"### ⚠️ REKOMENDASI: TOLAK (HIGH RISK)")
                    st.write(f"Probabilitas Gagal Bayar: **{proba_persen:.2f}%**")
                else:
                    st.success(f"### ✅ REKOMENDASI: TERIMA (LOW RISK)")
                    st.write(f"Probabilitas Gagal Bayar: **{proba_persen:.2f}%**")
                
                st.progress(prob)

            except Exception as e:
                st.warning(f"Terjadi kesalahan teknis: {e}")