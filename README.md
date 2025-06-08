# Proyek Analisis Data: Prediksi Status Kelulusan Siswa Jaya Jaya Institut - Berliantino

## Business Understanding

### Permasalahan Bisnis

Jaya Jaya Institut menghadapi tantangan dalam memantau dan memahami faktor-faktor yang memengaruhi performa akademik dan status kelulusan siswa mereka. Dengan jumlah siswa yang terus bertambah, diperlukan pendekatan berbasis data untuk mengidentifikasi siswa yang berisiko dropout atau gagal, serta memahami pola yang berkontribusi terhadap keberhasilan akademik.

Pertanyaan bisnis utama yang perlu dijawab:
- Faktor apa saja yang paling signifikan memengaruhi status kelulusan siswa?
- Bagaimana kita dapat secara proaktif mengidentifikasi siswa yang berisiko dropout?

### Cakupan Proyek

Proyek ini bertujuan untuk membantu Jaya Jaya Institut dalam memahami data siswa dan memonitor performa mereka melalui:

1. Analisis data siswa (dengan label kategori bermakna) untuk mengidentifikasi pola dan faktor kunci yang memengaruhi status kelulusan (Lulus, Dropout, Terdaftar).
2. Pembuatan dashboard interaktif untuk memvisualisasikan data siswa dan metrik performa utama.
3. Pengembangan model machine learning yang dapat memprediksi status kelulusan siswa.
4. Penyediaan prototype aplikasi berbasis web (menggunakan Streamlit) agar solusi machine learning mudah diakses.
5. Penyusunan rekomendasi action items berdasarkan temuan analisis data.

### Persiapan

#### Sumber data:

Dataset yang digunakan dalam proyek ini adalah data siswa Jaya Jaya Institut yang mencakup 4424 records dengan 37 fitur, meliputi informasi demografis, akademik, sosial-ekonomi, dan status kelulusan siswa. Dataset asli (`data.csv`) menggunakan kode numerik untuk merepresentasikan nilai pada kolom kategorikal, yang kemudian dikonversi menjadi label teks bermakna (`data_labeled.csv`) untuk meningkatkan interpretasi.

Proses konversi data kategori:
1. **Ekstraksi Kamus Data:** Mapping dari kode numerik ke label teks diekstrak dari dokumentasi dataset.
2. **Skrip Konversi:** Skrip Python (`convert_data.py`) dibuat untuk mengkonversi kode numerik menjadi label teks bermakna.
3. **Dataset Hasil:** File `data_labeled.csv` berisi data yang sama dengan `data.csv`, namun dengan label teks yang bermakna pada kolom-kolom kategorikal.

Contoh mapping yang digunakan:
* `Marital_status`: `{1: 'single', 2: 'married', ...}`
* `Gender`: `{1: 'male', 0: 'female'}`
* `Debtor`: `{1: 'yes', 0: 'no'}`
* `Course`: `{33: 'Biofuel Production Technologies', 171: 'Animation and Multimedia Design', ...}`

#### Setup environment:

Proyek ini menggunakan Python 3.11 dengan library berikut:
- pandas==2.2.3
- scikit-learn==1.6.1
- joblib==1.5.1
- streamlit==1.45.1
- matplotlib
- seaborn
- numpy

Semua dependensi tercantum dalam file `requirements.txt` dan dapat diinstal dengan perintah:
```
pip install -r requirements.txt
```

## Business Dashboard

Dashboard interaktif telah dibuat untuk memvisualisasikan data siswa dan metrik performa utama, membantu stakeholder dalam pemantauan dan pengambilan keputusan.

### Link Dashboard

Dashboard dapat diakses pada tautan berikut: [Link Dashboard](https://lookerstudio.google.com/s/rfmeaukm6N8)

### Fitur Dashboard

Dashboard ini menyajikan:
1. **Metrik Kunci (KPI):**
   - Jumlah Total Siswa
   - Tingkat Kelulusan (% Graduate)
   - Tingkat Dropout (% Dropout)
   - Tingkat Pendaftaran Aktif (% Enrolled)
   - Rata-rata Usia Pendaftaran, Nilai Masuk
   - % Penerima Beasiswa, % Siswa dengan Tunggakan

2. **Visualisasi Utama:**
   - Distribusi Status Siswa (Pie/Bar Chart)
   - Status vs Faktor Kunci (Stacked/Grouped Bar Chart)
   - Distribusi Nilai (Histogram/Box Plot)
   - Nilai Akademik vs Status (Box Plot)
   - Status Keuangan vs Status (Stacked Bar Chart)
   - Tabel Detail dengan Filter

3. **Interaktivitas:**
   - Filter berdasarkan Course, Gender, Scholarship Holder, dll.
   - Drill-down untuk analisis lebih mendalam

## Menjalankan Sistem Machine Learning

Sistem machine learning untuk prediksi status kelulusan siswa telah diimplementasikan dalam bentuk aplikasi web interaktif menggunakan Streamlit.

### Aplikasi Streamlit

**URL:** [https://bme-graduation-prediction.streamlit.app/](https://bme-graduation-prediction.streamlit.app/)

### Overview Aplikasi

Aplikasi web berbasis Streamlit untuk memprediksi status kelulusan siswa (Graduate, Dropout, Enrolled) menggunakan machine learning di Jaya Jaya Institut.

#### Fitur Utama

**Dua Mode Prediksi:**
- **Input Manual:** Form interaktif dengan 7 tab kategori (Info Dasar, Akademik, Keluarga, Keuangan, Performa, Aplikasi, Lainnya)
- **Upload CSV:** Batch prediction dengan template download dan validasi otomatis

**Visualisasi Hasil:**
- **Single Prediction:** Status dengan confidence score, bar chart probabilitas, gauge indicator
- **Batch Prediction:** Summary metrics, tabel hasil, pie chart distribusi, download CSV

### Cara Menjalankan Aplikasi Secara Lokal

1. **Clone repository:**
   ```
   git clone https://github.com/berliantino/bme-graduation-prediction.git
   cd <repository-directory>
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**
   ```
   streamlit run app.py
   ```

4. **Akses aplikasi:**
   Buka browser dan akses `http://localhost:8501`

### Deployment ke Streamlit Cloud

1. **Siapkan Akun:** GitHub & Streamlit Community Cloud.
2. **Buat Repository GitHub:** Unggah semua file dari proyek ini.
3. **Hubungkan ke Streamlit Cloud:** Login ke Streamlit Cloud, pilih "New app" > "From existing repo", pilih repo Anda, pastikan branch dan path file utama (`app.py`) benar, lalu klik "Deploy!".
4. **Akses Aplikasi:** Gunakan URL publik yang disediakan setelah deployment selesai.

## Conclusion

Proyek ini berhasil menganalisis data siswa Jaya Jaya Institut menggunakan label kategori yang bermakna, mengidentifikasi faktor kunci yang memengaruhi status kelulusan, dan mengembangkan solusi prediktif yang dapat digunakan untuk intervensi dini.

Hasil analisis menunjukkan bahwa:
1. Faktor keuangan (status tunggakan, pembayaran biaya kuliah tepat waktu, dan status beasiswa) memiliki korelasi kuat dengan status kelulusan.
2. Performa akademik pada semester awal sangat menentukan kemungkinan kelulusan.
3. Model machine learning berbasis Random Forest mencapai akurasi ~77% dalam memprediksi status kelulusan siswa.

Kombinasi dashboard interaktif dan aplikasi prediksi memberikan alat yang komprehensif bagi Jaya Jaya Institut untuk memantau performa siswa dan mengidentifikasi siswa berisiko dropout secara proaktif.

## Rekomendasi Action Items

Berdasarkan hasil analisis, berikut rekomendasi tindakan konkret untuk Jaya Jaya Institut:

1. **Intervensi Dini:** Gunakan dashboard/aplikasi untuk identifikasi siswa berisiko (nilai rendah, `Debtor: yes`, `Tuition_fees_up_to_date: no`) dan berikan pendampingan akademik dan konseling.

2. **Dukungan Keuangan:** Perkuat program beasiswa (`Scholarship_holder: yes` berkorelasi positif dengan kelulusan) dan layanan konseling keuangan untuk siswa dengan kesulitan ekonomi.

3. **Pantau Kinerja Akademik Awal:** Fokus pada pemantauan dan dukungan akademik di semester 1 & 2, karena performa di periode ini sangat menentukan hasil akhir.

4. **Integrasi Alat:** Dorong penggunaan dashboard dan alat prediksi dalam proses operasional rutin institusi, termasuk pelatihan staf akademik dan administratif.

5. **Iterasi Model:** Kumpulkan data baru secara berkala, latih ulang model prediktif, dan eksplorasi teknik machine learning lain untuk meningkatkan akurasi prediksi.
