# Prediksi Average Playtime Game Steam Menggunakan Random Forest Regressor

Proyek *Machine Learning* ini bertujuan untuk memprediksi durasi rata-rata waktu bermain (*Average Playtime Forever*) pada game di platform **Steam** berdasarkan karakteristik teknis dan kategori game (seperti harga, jumlah DLC, achievement, batas usia, platform OS, genre, serta kategori gameplay).

---

## 📌 Identitas Proyek
- **Nama Pengembang** : Muhamad Fajar
- **NIM** : 241011400752
- **Mata Kuliah** : Machine Learning (Semester 5)
- **Topik Proyek** : Regresi Prediksi *Playtime* Game Steam & Model Persistence

---

## 📂 Struktur Direktori Proyek

```plaintext
steam-playtime-prediction/
│
├── games.csv                   # Dataset mentah game Steam
├── Machine_Learning.ipynb      # Notebook eksperimen ML (EDA, Preprocessing, Modeling, Evaluasi)
├── random_forest_model.pkl     # Model terlatih (Random Forest Regressor + StandardScaler + Metadata)
└── README.md                   # Dokumentasi lengkap proyek
```

### Penjelasan File:
1. **`games.csv`**: Dataset mentah yang berisi data ribuan judul game Steam beserta atribut seperti harga, genre, platform OS, rating, dan waktu bermain.
2. **`Machine_Learning.ipynb`**: Notebook utama yang memuat keseluruhan kode analisis, pembersihan data, visualisasi EDA, rekayasa fitur (*feature engineering*), pelatihan model, hingga evaluasi performa regresi.
3. **`random_forest_model.pkl`**: Berkas model biner hasil serialisasi (*pickle*) yang memuat objek estimator `RandomForestRegressor`, `StandardScaler` ter-fit, serta daftar nama fitur yang siap digunakan untuk inferensi.
4. **`README.md`**: Panduan dan dokumentasi teknis menyeluruh mengenai alur kerja dan penggunaan model.

---

## ⚙️ Tahapan & Alur Metodologi Machine Learning

```
[games.csv]
     │
     ▼
1. Data Cleaning & Transformation  ──► Imputasi/hapus missing value, perbaikan tipe data, pembersihan teks
     │
     ▼
2. Exploratory Data Analysis (EDA) ──► Analisis distribusi playtime, korelasi fitur numerik, tren genre
     │
     ▼
3. Feature Engineering & Splitting ──► StandardScaler (fitur numerik) + One-Hot Encoding (genre & mode)
     │                                 Train-Test Split (80% Train, 20% Test)
     ▼
4. Model Training & Evaluation     ──► Random Forest Regressor (100 Trees, Max Depth: 15)
     │                                 Evaluasi: MAE, RMSE, R² Score
     ▼
5. Model Persistence (Export)      ──► Serialisasi artefak ke random_forest_model.pkl
```

### 1. Data Cleaning & Transformation
- **Pembersihan Missing Values & Duplikasi**: Menangani nilai kosong pada kolom genre, kategori, dan deskripsi, serta menghapus rekaman data yang redundan.
- **Transformasi Tipe Data**: Mengonversi kolom numerik (*Price*, *DLC count*, *Achievements*, *Required age*) ke format float/integer.
- **Ekstraksi Fitur Platform**: Mengekstraksi kolom biner untuk sistem operasi: `Windows`, `Mac`, dan `Linux`.

### 2. Exploratory Data Analysis (EDA)
- Menganalisis distribusi variabel target (`Average playtime forever`), yang memiliki distribusi menceng ke kanan (*skewed*).
- Meneliti hubungan antara harga game (*Price*) dan ketersediaan *DLC* terhadap durasi pemain memainkan game.
- Mengidentifikasi genre-genre yang memiliki rata-rata retensi bermain paling lama (seperti RPG, Simulasi, dan Multiplayer).

### 3. Feature Engineering & Data Splitting
- **Standarisasi Numerik**: Menggunakan `StandardScaler` dari Scikit-Learn pada fitur `Price`, `DLC count`, `Achievements`, dan `Required age` agar memiliki skala rata-rata 0 dan variansi 1.
- **One-Hot Encoding**:
  - Mengambil **Top 10 Genre terpopuler**: *Indie*, *Action*, *Adventure*, *Casual*, *Simulation*, *RPG*, *Strategy*, *Early Access*, *Free To Play*, dan *Sports*.
  - Mengambil **Kategori Gameplay utama**: *Single-player*, *Multi-player*, *Co-op*, *Steam Achievements*, dan *Steam Cloud*.
- **Data Splitting**: Membagi dataset menjadi data latih (*training set*) dan data uji (*test set*) dengan rasio **80% : 20%**.

### 4. Pelatihan & Evaluasi Model
- Model dilatih menggunakan algoritma ensemble **Random Forest Regressor** dengan konfigurasi:
  - `n_estimators`: 100 pohon keputusan (*decision trees*)
  - `max_depth`: 15 (untuk mengontrol kompleksitas dan mencegah *overfitting*)
  - `random_state`: 42 (menjamin reproduktifitas eksperimen)
- Evaluasi performa model diuji menggunakan metrik regresi:
  - **MAE (Mean Absolute Error)**: Mengukur rata-rata deviasi absolut prediksi terhadap nilai aktual.
  - **RMSE (Root Mean Squared Error)**: Mengukur deviasi prediksi dengan bobot penalti lebih besar pada error yang ekstrem.
  - **R² Score (Koefisien Determinasi)**: Mengukur proporsi variansi variabel target yang berhasil dijelaskan oleh model.

---

## 📊 Daftar 22 Fitur Input Model

Model menggunakan **22 fitur** input yang telah ditransformasi:

| No | Nama Kolom Fitur | Tipe Data | Deskripsi |
|:---:|---|:---:|---|
| 1 | `Required age` | Float (Scaled) | Batas usia minimum pemain |
| 2 | `Price` | Float (Scaled) | Harga game dalam satuan USD |
| 3 | `DLC count` | Float (Scaled) | Jumlah konten tambahan (DLC) |
| 4 | `Windows` | Integer (0/1) | Dukungan untuk sistem operasi Windows |
| 5 | `Mac` | Integer (0/1) | Dukungan untuk sistem operasi MacOS |
| 6 | `Linux` | Integer (0/1) | Dukungan untuk sistem operasi Linux |
| 7 | `Achievements` | Float (Scaled) | Jumlah achievement/prestasi dalam game |
| 8 | `Genre_Indie` | Integer (0/1) | Game memiliki genre Indie |
| 9 | `Genre_Action` | Integer (0/1) | Game memiliki genre Action |
| 10 | `Genre_Adventure` | Integer (0/1) | Game memiliki genre Adventure |
| 11 | `Genre_Casual` | Integer (0/1) | Game memiliki genre Casual |
| 12 | `Genre_Simulation` | Integer (0/1) | Game memiliki genre Simulation |
| 13 | `Genre_RPG` | Integer (0/1) | Game memiliki genre RPG (Role-Playing Game) |
| 14 | `Genre_Strategy` | Integer (0/1) | Game memiliki genre Strategy |
| 15 | `Genre_Early Access` | Integer (0/1) | Game berstatus Early Access |
| 16 | `Genre_Free To Play` | Integer (0/1) | Game bersifat Free To Play (Gratis) |
| 17 | `Genre_Sports` | Integer (0/1) | Game memiliki genre Sports |
| 18 | `Mode_Single-player` | Integer (0/1) | Mendukung mode bermain Single-player |
| 19 | `Mode_Multi-player` | Integer (0/1) | Mendukung mode bermain Multi-player |
| 20 | `Mode_Co-op` | Integer (0/1) | Mendukung mode bermain Co-op |
| 21 | `Mode_Steam Achievements` | Integer (0/1) | Terintegrasi dengan Steam Achievements |
| 22 | `Mode_Steam Cloud` | Integer (0/1) | Terintegrasi dengan penyimpanan Steam Cloud |

---

## 📦 Isi Bundel Model (`random_forest_model.pkl`)

File `random_forest_model.pkl` menyimpan dictionary artefak lengkap agar proses inferensi pada data baru dapat dilakukan tanpa kehilangan konteks preprocessing:

```python
{
    'model': <RandomForestRegressor>,            # Objek model yang sudah di-training
    'scaler': <StandardScaler>,                  # Objek scaler yang sudah di-fit pada data train
    'features_to_scale': [...],                  # ['Price', 'DLC count', 'Achievements', 'Required age']
    'feature_columns': [...],                    # Urutan 22 kolom fitur yang diharapkan model
    'top_10_genres': [...],                      # Daftar 10 genre utama
    'available_modes': [...]                     # Daftar mode gameplay utama
}
```

---

## 💻 Panduan Menjalankan & Memuat Model

### 1. Prasyarat Instalasi
Pastikan dependensi Python telah terpasang:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Membuka Notebook Eksperimen
Jalankan Jupyter Notebook untuk melihat seluruh proses analitis:
```bash
jupyter notebook Machine_Learning.ipynb
```

### 3. Menggunakan Model untuk Prediksi (Inference via Script Python)
Anda dapat memuat model dan melakukan prediksi pada data game baru menggunakan kode Python berikut:

```python
import pickle
import pandas as pd
import numpy as np

# 1. Memuat bundel model
with open('random_forest_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
scale_cols = bundle['features_to_scale']
feat_cols = bundle['feature_columns']

# 2. Contoh data game baru
new_game = {
    'Required age': 0,
    'Price': 19.99,
    'DLC count': 2,
    'Windows': 1,
    'Mac': 1,
    'Linux': 0,
    'Achievements': 30,
    'Genre_Indie': 0,
    'Genre_Action': 1,
    'Genre_Adventure': 1,
    'Genre_Casual': 0,
    'Genre_Simulation': 0,
    'Genre_RPG': 1,
    'Genre_Strategy': 0,
    'Genre_Early Access': 0,
    'Genre_Free To Play': 0,
    'Genre_Sports': 0,
    'Mode_Single-player': 1,
    'Mode_Multi-player': 1,
    'Mode_Co-op': 0,
    'Mode_Steam Achievements': 1,
    'Mode_Steam Cloud': 1
}

# 3. Scaling fitur numerik
num_df = pd.DataFrame([{col: new_game[col] for col in scale_cols}])
scaled_values = scaler.transform(num_df)[0]
for idx, col in enumerate(scale_cols):
    new_game[col] = scaled_values[idx]

# 4. Melakukan prediksi
input_df = pd.DataFrame([new_game])[feat_cols]
pred_minutes = model.predict(input_df)[0]

print(f"Prediksi Average Playtime: {pred_minutes:.2f} menit ({pred_minutes/60:.2f} jam)")
```

---

## 🎓 Kesimpulan

Proyek ini telah berhasil mengimplementasikan alur kerja pemodelan regresi Machine Learning menggunakan algoritma **Random Forest Regressor** pada dataset game Steam. Dengan melakukan serialisasi model beserta seluruh artefak *preprocessing* ke dalam file `random_forest_model.pkl`, model siap diintegrasikan atau digunakan untuk analisis prediktif lebih lanjut secara konsisten dan terstandardisasi.
