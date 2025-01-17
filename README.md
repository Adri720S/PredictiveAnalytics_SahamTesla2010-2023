# **Laporan Proyek Machine Learning - Adri Sopiana**

## **Domain Proyek**

**Latar Belakang:**

Pasar saham merupakan salah satu elemen paling dinamis dalam dunia ekonomi dan bisnis. Pergerakan harga saham dapat memberikan wawasan yang sangat berguna bagi investor untuk membuat keputusan investasi yang tepat. Dalam proyek ini, data historis harga saham Tesla (2010-2023) digunakan untuk menganalisis tren dan memprediksi harga saham di masa depan.

**Mengapa masalah ini penting?**
- Investor memerlukan alat yang dapat membantu memprediksi pergerakan saham untuk mengurangi risiko investasi.
- Perusahaan Tesla adalah salah satu perusahaan teknologi terkemuka dengan volatilitas saham yang menarik perhatian dunia.

**Referensi:**
- [Stock Market Prediction using LSTM](https://scholar.google.com/)
- [Time Series Analysis for Stock Price Prediction](https://scholar.google.com/)

## **Business Understanding**

### **Problem Statements**
1. Bagaimana menganalisis tren historis harga saham Tesla untuk memahami pola pergerakannya?
2. Bagaimana cara memprediksi harga saham Tesla di masa depan berdasarkan data historis?

### **Goals**
1. Memberikan wawasan kepada investor tentang pola pergerakan harga saham Tesla.
2. Membuat model prediktif untuk memperkirakan harga saham Tesla pada periode mendatang.

### **Solution Statements**
1. Menggunakan model **LSTM (Long Short-Term Memory)** untuk memprediksi harga saham, karena LSTM dikenal efektif untuk menangani data time series.
2. Melakukan hyperparameter tuning untuk meningkatkan kinerja model.

## **Data Understanding**

### **Informasi Dataset**
Kumpulan data berisi harga saham historis untuk perusahaan Tesla dari tahun 2010 sampai tahun 2023 dengan jumlah data sebanyak 3162 baris dan 7 kolom. Informasi ini dapat digunakan untuk menganalisis kinerja saham perusahaan dari waktu ke waktu dan membuat prediksi tentang kinerja di masa mendatang. Informasi ini dapat digunakan untuk mempelajari tren dan pola di pasar saham dan membuat keputusan investasi yang tepat. Dataset ini mencakup fitur-fitur berikut:
- **Date:** Tanggal transaksi saham.
- **Open:** Harga pembukaan saham pada tanggal tersebut.
- **High:** Harga tertinggi saham pada tanggal tersebut.
- **Low:** Harga terendah saham pada tanggal tersebut.
- **Close:** Harga penutupan saham pada tanggal tersebut (digunakan sebagai target).
- **Adj Close:** Harga penutupan yang disesuaikan.
- **Volume:** Jumlah saham yang diperdagangkan pada tanggal tersebut.

**1. Tipe Data**:
Tipe data terdiri dari float64 untuk nilai harga saham, int64 untuk volume, dan object untuk tanggal.

**2. Penanganan Missing Value**:
Penanganan missing value menggunakan metode forward fill yaitu mengisi nilai hilang dengan nilai rata-rata atau nilai sebelumnya. Setelah dianalisis kondisi dataset tidak ada missing value.
     
**3. Duplicate Data**:
Data duplikat diidentifikasi menggunakan df.duplicated().sum(). Hasilnya kondisi dataset tidak ada duplikat data.

**4. Statistik Deskriptif**:
Ringkasan statistik numerik ditampilkan menggunakan .describe().

**5. Outlier Detection**:
- Menggunakan metode IQR untuk mendeteksi outlier pada setiap kolom numerik.
- Visualisasi distribusi menggunakan boxplot menunjukkan bahwa outlier adalah bagian alami dari data saham.

**Sumber Data:** (https://www.kaggle.com/datasets/muhammadbilalhaneef/-tesla-stock-price-from-2010-to-2023?resource=download) 

## **Data Preparation**

### **Tahapan Data Preparation**
**1. Mengubah kolom Date menjadi tipe datetime**:
Tujuan kolom Date diubah menjadi tipe datetime agar dapat membantu memastikan data diurutkan berdasarkan waktu atau data dapat diolah secara kronologis, yang merupakan syarat utama untuk analisis deret waktu (time-series). 

**2. Normalisasi Data**:
LSTM lebih efektif jika data input berada dalam skala yang seragam maka data dinormalisasi menggunakan **MinMaxScaler** agar nilai berada pada rentang [0,1], yang diperlukan untuk model.

**3. Split Data**:
Dataset dibagi menjadi 80% data training dan 20% data testing.

**4. Membuat Sequence Data**:
LSTM membutuhkan data dalam bentuk **urutan** untuk belajar pola dari data sebelumnya (temporal dependencies). Sehingga sequence data dibuat dengan menggunakan 60 hari terakhir sebagai input untuk memprediksi harga hari ke-61.

**5. Visualisasi Data**:
Pergerakan harga penutupan saham Tesla divisualisasikan menggunakan line plot. Agar gambaran umum tentang pola harga saham selama periode waktu tertentu dapat terlihat.

![alt text](https://github.com/Adri720S/PredictiveAnalytics_SahamTesla2010-2023/blob/main/download%20(1).png?raw=true)

## **Modeling**

### **Model yang Digunakan**
1. **LSTM**:
   - Digunakan karena mampu menangani data time series dengan baik, terutama dalam mendeteksi pola jangka panjang dan pendek.

2. **Arsitektur Model**:
   - 2 lapisan LSTM dengan masing-masing 50 unit.
   - Dropout untuk mencegah overfitting.
   - Dense untuk menghasilkan output akhir berupa harga prediksi.
  
3. **Parameter Training**:
   - Batch size: 32
   - Epochs: 50
   - Optimizer: Adam
   - Loss Function: Mean Squared Error

4. **Implementasi Model LSTM**:
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense, Dropout

   model = Sequential([
       LSTM(50, return_sequences=True, input_shape=(60, 1)),
       Dropout(0.2),
       LSTM(50, return_sequences=False),
       Dropout(0.2),
       Dense(25),
       Dense(1)
   ])
   model.compile(optimizer='adam', loss='mean_squared_error')
   model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))
   ```

## **Evaluation**

### **Metrik Evaluasi**
**a. Root Mean Squared Error (RMSE):**

Metrik ini mengukur seberapa jauh prediksi model dari nilai sebenarnya.
     
**b. Hasil Evaluasi**:

**RMSE Model LSTM:** 14.99
     
**c. Dampak Model terhadap Business Understanding**

Model yang dibangun menggunakan LSTM memberikan kontribusi terhadap Business Understanding sebagai berikut:

1. Analisis Tren Historis
   Dengan mempelajari data historis harga saham Tesla, model memberikan wawasan tentang pola pergerakan harga saham, seperti:
   - Tren kenaikan atau penurunan dalam jangka waktu tertentu.
   - Pola musiman atau fluktuasi harga.
   - Periode volatilitas tinggi atau rendah. Wawasan ini dapat digunakan oleh investor untuk membuat keputusan investasi yang lebih bijak.
  
2. Prediksi Harga Saham di Masa Depan
Prediksi yang dihasilkan oleh model memberikan estimasi harga saham Tesla pada periode mendatang. Walaupun tidak sepenuhnya akurat (karena sifat pasar yang dinamis), prediksi ini dapat digunakan sebagai acuan oleh investor untuk memperkirakan peluang dan risiko dalam berinvestasi.

**d. Evaluasi Problem Statements**

1. Model LSTM telah memanfaatkan data historis (harga saham 60 hari sebelumnya) untuk mengidentifikasi pola yang relevan. Melalui visualisasi hasil, tren historis dapat dilihat dengan jelas, termasuk bagaimana pola harga dipelajari oleh model.

2. Model LSTM mampu memprediksi harga saham Tesla berdasarkan pola yang dipelajari dari data historis. Prediksi diuji menggunakan data testing (20% dari total dataset), menghasilkan error yang cukup rendah (dengan metrik RMSE).

**e. Evaluasi Goals**

1. Analisis tren historis yang dilakukan memberikan wawasan tentang pola pergerakan harga saham, seperti periode kenaikan, penurunan, atau fluktuasi besar. Grafik perbandingan antara harga aktual dan harga prediksi membantu investor memahami bagaimana data historis memengaruhi prediksi di masa depan.

2. Model prediktif berbasis LSTM telah berhasil dibuat dan diuji. Model dapat memberikan prediksi harga saham untuk periode tertentu, yang dapat digunakan oleh investor sebagai referensi tambahan.

**f. Evaluasi Solution Statements**

1. LSTM terbukti mampu menangani data time series dengan baik, karena memanfaatkan kemampuan untuk mengingat informasi jangka panjang. Prediksi harga saham Tesla yang dihasilkan mendekati harga aktual, seperti yang terlihat dalam grafik hasil prediksi.

2. Hyperparameter tuning (seperti jumlah unit LSTM, jumlah epoch, dan learning rate) membantu meningkatkan performa model dengan menurunkan nilai error (RMSE). Proses tuning menghasilkan model yang lebih optimal dibandingkan pengaturan default.

**g. Visualisasi Hasil**

Berikut adalah visualisasi perbandingan antara harga saham aktual Tesla dan harga prediksi yang dihasilkan oleh model LSTM:

![alt text](https://github.com/Adri720S/PredictiveAnalytics_SahamTesla2010-2023/blob/main/download.png?raw=true)

Grafik ini menunjukkan performa model dalam memprediksi harga saham Tesla berdasarkan data testing.

## **Kesimpulan**
- Model LSTM menunjukkan performa yang jauh lebih baik dibandingkan baseline regresi linier untuk memprediksi harga saham Tesla, dengan RMSE yang lebih rendah.
- Model ini mampu menangkap pola time series dengan baik, tetapi masih ada ruang untuk perbaikan, misalnya dengan mencoba model deep learning lain seperti GRU atau menggunakan lebih banyak fitur tambahan.
