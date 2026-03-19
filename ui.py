import streamlit as st
import requests

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Internal Credit Risk System", layout="wide")

# 2. HEADER & VISUAL
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.title("🏦") 
with col_text:
    st.title("Credit Risk Assessment Dashboard")
    st.write("**Internal Tools for Credit Officers & Analysts**")

st.image("https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
         caption="Sistem Prediksi Probabilitas Gagal Bayar (Default Probability)", use_container_width=True)

# 3. PANDUAN PENGGUNA
with st.expander("📖 Panduan Operasional & Referensi Parameter (KLIK DI SINI)"):
    st.markdown("""
    ### **Siapa yang Menggunakan Aplikasi Ini?**
    Aplikasi ini dirancang khusus untuk **Petugas Kredit (Credit Officer)** sebagai sistem pendukung keputusan (DSS).
    
    ### **Pedoman Penentuan Parameter Internal:**
    Gunakan referensi standar berikut untuk pengisian data bank:
    
    | Loan Grade | Suku Bunga (%) | Keterangan Risiko |
    | :--- | :--- | :--- |
    | **A** | 5.0% - 9.0% | Nasabah sangat aman, riwayat kredit sempurna. |
    | **B** | 9.1% - 12.0% | Nasabah aman, riwayat kredit stabil. |
    | **C** | 12.1% - 15.0% | Risiko menengah, ada keterlambatan kecil di masa lalu. |
    | **D - G** | > 15.0% | Risiko tinggi, riwayat kredit buruk/kurang lancar. |

    ### **Logika Bisnis:**
    * **Batas Usia:** 18 - 65 tahun (Masa Produktif).
    * **Threshold:** 0.32 (Prediksi di atas 32% dianggap risiko tinggi/gagal bayar).
    """)

st.divider()

# 4. INPUT USER
st.subheader("📝 Data Pengajuan Kredit")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### **🔵 Identitas & Kemampuan Nasabah**")
    st.caption("Diisi berdasarkan dokumen resmi nasabah")
    person_age = st.number_input("Usia Nasabah (Tahun):", 18, 80, 25)
    person_income = st.number_input("Pendapatan Tahunan (USD):", 1000, 1000000, 50000)
    person_emp_length = st.number_input("Lama Bekerja (Tahun):", 0.0, 50.0, 2.0)
    person_home_ownership = st.selectbox("Status Kepemilikan Rumah:", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (Tahun):", 0, 40, 3)

with col2:
    st.markdown("### **🔴 Parameter & Kebijakan Internal Bank**")
    st.caption("Diisi oleh Petugas Bank berdasarkan hasil scoring awal")
    loan_intent = st.selectbox("Tujuan Pinjaman:", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    
    loan_grade = st.selectbox("Loan Grade (Hasil Scoring Awal):", 
                              ["A", "B", "C", "D", "E", "F", "G"],
                              help="A = Terbaik, G = Terburuk. Ditentukan dari skor kredit nasabah.")
    
    loan_amnt = st.number_input("Jumlah Pinjaman yang Diajukan (USD):", 500, 50000, 5000)
    
    loan_int_rate = st.number_input("Suku Bunga Penawaran (%):", 0.1, 25.0, 11.0,
                                   help="Semakin tinggi Grade, biasanya suku bunga semakin rendah.")
    
    cb_person_default_on_file = st.selectbox("Pernah Gagal Bayar Sebelumnya (Data SLIK)?", ["N", "Y"])

# Hitung Rasio Utang (DTI) Otomatis
loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
st.metric("Debt to Income Ratio (DTI)", f"{loan_percent_income}", help="Rasio jumlah pinjaman dibandingkan pendapatan.")

# 5. PROSES PREDIKSI 
if st.button("🚀 Jalankan Prediksi"):
    
    # --- JARING PENGAMAN (VALIDASI BISNIS) ---
    if person_age > 65:
        st.error("❌ **AKSES DITOLAK:** Nasabah berada di luar usia produktif (> 65 tahun). Sesuai kebijakan, pengajuan tidak dapat diproses.")
    
    elif person_income < 5000:
        st.error("❌ **AKSES DITOLAK:** Pendapatan tahunan tidak memenuhi batas minimum ($5,000).")
        
    elif loan_percent_income > 0.60:
        st.warning("⚠️ **RISIKO KRITIS:** Rasio utang (DTI) melebihi 60%. Nasabah sangat rentan gagal bayar.")
        
    else:
        # Menyiapkan data JSON untuk dikirim ke API
        data_input = {
            "person_age": int(person_age),
            "person_income": int(person_income),
            "person_home_ownership": person_home_ownership,
            "person_emp_length": float(person_emp_length),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": int(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": int(cb_person_cred_hist_length)
        }

        with st.spinner("🤖 AI sedang menganalisis probabilitas risiko..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=data_input)
                
                if response.status_code == 200:
                    result = response.json()
                    st.divider()
                    
                    proba_persen = result['prediction_proba'] * 100
                    
                    if result["prediction_class"] == 1:
                        st.error("### ⚠️ REKOMENDASI: TOLAK (HIGH RISK)")
                        st.write(f"Probabilitas Gagal Bayar: **{proba_persen:.2f}%** (Kategori: Risiko Tinggi)")
                        st.info(f"💡 **Analisis AI:** Skor risiko ({proba_persen:.2f}%) melebihi ambang batas 32%.")
                    else:
                        if proba_persen < 10: kategori = "Sangat Aman (Low Risk)"
                        elif proba_persen < 20: kategori = "Aman (Moderate-Low Risk)"
                        else: kategori = "Waspada (Moderate Risk)"
                            
                        st.success("### ✅ REKOMENDASI: TERIMA (LOW RISK)")
                        st.write(f"Probabilitas Gagal Bayar: **{proba_persen:.2f}%** (Kategori: {kategori})")
                        st.info(f"💡 **Analisis AI:** Skor risiko ({proba_persen:.2f}%) di bawah ambang batas 32%.")
                    
                    st.progress(result['prediction_proba'])
                    
                else:
                    st.error(f"API Error: Status {response.status_code}.")
                    
            except Exception as e:
                st.error(f"Koneksi Gagal! Pastikan file api.py sudah dijalankan.")