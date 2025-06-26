# Prediksi Konsumsi Listrik Rumah Tangga

Proyek ini bertujuan untuk memprediksi konsumsi listrik rumah tangga berdasarkan data aktivitas harian menggunakan teknik Machine Learning.

## 📁 Struktur Proyek

```bash
.
├── data/
│   ├── raw/                # Data mentah (jika ada)
│   ├── processed/          # Data yang sudah dibersihkan dan diolah
│   └── generated/          # Dataset hasil generate otomatis (CSV)
├── models/                 # Model hasil training (pkl)
├── reports/
│   ├── figures/            # Visualisasi seperti plot evaluasi & feature importance
│   └── final_report.md     # Laporan akhir
├── src/
│   ├── data_processing.py       # Load & preprocessing data
│   ├── feature_engineering.py   # Analisis korelasi & feature importance
│   ├── models.py                # Training, evaluasi & tuning model
│   └── evaluation.py            # Evaluasi akhir & visualisasi hasil
├── notebooks/             # Notebook Jupyter untuk dokumentasi setiap tahapan proyek
│   ├── 01_data_exploration.ipynb     # Eksplorasi data
│   ├── 02_data_preparation.ipynb     # Persiapan dan preprocessing
│   ├── 03_modeling.ipynb             # Proses training dan evaluasi model
│   └── 04_evaluation.ipynb           # Evaluasi akhir & visualisasi
├── main.py                 # Main pipeline script
├── requirements.txt        # Daftar dependensi
└── README.md               # Dokumentasi proyek ini
```

## 📊 Dataset

Dataset disimpan di `data/generated/listrik_rumah.csv`, dan akan digenerate secara otomatis jika belum tersedia menggunakan `generate_dataset.py`.

## ⚙️ Cara Menjalankan

1. **Clone repositori** ini ke lokal:

   ```bash
   git clone https://github.com/danskuy13/Household-Electricity-Consumption-Prediction-Based-on-Daily-Activities.git
   cd electric-usage-prediction
   ```

2. **Buat environment virtual dan install dependensi**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan main.py**:

   ```bash
   python main.py
   ```

4. **Hasil akhir** akan berupa:

   - Model tersimpan di `models/`
   - Gambar evaluasi di `reports/figures/`
   - Laporan akhir di `reports/final_report.md`

## 📦 Dependensi

Lihat `requirements.txt`. Versi Python yang direkomendasikan: **Python 3.10+**.

## 📈 Output Visualisasi

Beberapa visualisasi yang dihasilkan:

- Korelasi antar fitur (`correlation_matrix.png`)
- Perbandingan performa model (`model_comparison.png`)
- Feature importance model terbaik (`feature_importance_*.png`)
- Plot prediksi vs aktual (`actual_vs_predicted_...png`)
- Learning curve (`learning_curve_*.png`)

## 🧪 Notebook Jupyter

Notebook Jupyter digunakan untuk dokumentasi setiap tahapan proyek. Masing-masing notebook memiliki dokumentasi dalam bentuk **text cell** untuk menjelaskan proses dan hasil.

- `01_data_exploration.ipynb`: Eksplorasi awal data dan visualisasi korelasi
- `02_data_preparation.ipynb`: Proses pembersihan dan transformasi fitur
- `03_modeling.ipynb`: Training, evaluasi, dan tuning model
- `04_evaluation.ipynb`: Analisis performa akhir model terbaik

## ✍️ Author

- Nama: [Dani Ramdani]
- Tahun: 2025

---

Proyek ini disusun sebagai bagian dari studi tentang penerapan Machine Learning untuk efisiensi energi rumah tangga.
