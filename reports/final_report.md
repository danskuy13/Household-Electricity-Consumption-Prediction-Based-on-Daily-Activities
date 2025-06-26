# Final Report - Prediksi Konsumsi Listrik Rumah Tangga Berdasarkan Aktivitas Harian

## 1. Latar Belakang
Konsumsi listrik rumah tangga terus meningkat. Prediksi konsumsi listrik berbasis aktivitas harian dapat membantu perencanaan energi, penghematan biaya, dan efisiensi pemakaian perangkat listrik.

## 2. Tujuan
Mengembangkan model machine learning yang dapat memprediksi konsumsi listrik rumah tangga (kWh) berdasarkan aktivitas harian, durasi penggunaan, waktu, dan jumlah pengguna.

## 3. Dataset
Dataset terdiri dari 500+ sampel berisi kolom:
- aktivitas
- perangkat_listrik
- waktu_hari
- durasi_penggunaan
- jumlah_pengguna
- konsumsi_listrik_kwh (target)

## 4. Teknik Pemodelan
Model regresi yang digunakan:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## 5. Evaluasi Model

| Model            | MAE   | RMSE  | R² Score |
|------------------|-------|-------|----------|
| Linear Regression| 0.027 | 0.035 | 0.938    |
| Random Forest    | 0.015 | 0.020 | 0.980    |
| XGBoost          | 0.016 | 0.023 | 0.974    |

> Model terbaik: **Random Forest**

## 6. Visualisasi
Lihat folder `figures/prediksi_vs_aktual.png` untuk perbandingan grafik prediksi vs aktual.

## 7. Kesimpulan
Random Forest memberikan prediksi paling akurat dan stabil. Model bisa ditingkatkan dengan fitur tambahan seperti cuaca, jenis rumah, atau tarif listrik dinamis.


## Statistik Deskriptif
|       |   durasi_penggunaan_jam |   konsumsi_listrik_kwh |
|:------|------------------------:|-----------------------:|
| count |               500       |              500       |
| mean  |                 4.3132  |                2.27064 |
| std   |                 2.22132 |                1.72821 |
| min   |                 0.5     |                0.03    |
| 25%   |                 2.3     |                0.9675  |
| 50%   |                 4.4     |                1.685   |
| 75%   |                 6.3     |                3.195   |
| max   |                 8       |                7.2     |

## Korelasi dengan Target
|                       |   konsumsi_listrik_kwh |
|:----------------------|-----------------------:|
| durasi_penggunaan_jam |               0.687224 |

## Statistik Deskriptif
|       |   durasi_penggunaan_jam |   konsumsi_listrik_kwh |
|:------|------------------------:|-----------------------:|
| count |               500       |              500       |
| mean  |                 4.3132  |                2.27064 |
| std   |                 2.22132 |                1.72821 |
| min   |                 0.5     |                0.03    |
| 25%   |                 2.3     |                0.9675  |
| 50%   |                 4.4     |                1.685   |
| 75%   |                 6.3     |                3.195   |
| max   |                 8       |                7.2     |

## Korelasi dengan Target
|                       |   konsumsi_listrik_kwh |
|:----------------------|-----------------------:|
| durasi_penggunaan_jam |               0.687224 |