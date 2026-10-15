# 🤖 UKSW AI Studio - Web Machine Learning & Cloud Database

Aplikasi Web Machine Learning interaktif berbasis **Streamlit** dan terintegrasi dengan **MongoDB Atlas** untuk Ujian Akhir Semester (TAS) Kecerdasan Buatan - Universitas Kristen Satya Wacana (UKSW).

**Dosen Pengampu:** Dr. Suryasatriya Trihandaru, M.Sc.nat  
**Deadline:** 1 Desember 2026  

---

## 🚀 Fitur Unggulan

Aplikasi ini memenuhi 100% persyaratan spesifikasi TAS:

1. **📈 Regresi Linear & Non-Linear**:
   - Model: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest Regressor.
   - Evaluasi Metrik: MAE, MSE, RMSE, dan $R^2$ Score.
   - Visualisasi: Plot interaktif *Actual vs Predicted* dengan garis referensi ideal ($y = x$).

2. **🎯 Klasifikasi Multikelas & Biner**:
   - Model: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM).
   - Evaluasi: Accuracy, Precision, Recall, F1-Score, dan *Classification Report*.
   - Visualisasi: *Heatmap Confusion Matrix* interaktif.

3. **🔮 Clustering (Unsupervised Learning)**:
   - Algoritma: K-Means Clustering, Agglomerative (Hierarchical), DBSCAN.
   - Evaluasi: *Silhouette Score* dan *Davies-Bouldin Index*.
   - Visualisasi: Proyeksi klaster 2D interaktif berbasis *Principal Component Analysis (PCA)*.

4. **🍃 Integrasi MongoDB Atlas & Retraining Real-Time**:
   - Manajemen koneksi aman via `st.secrets["MONGODB_URI"]` dengan cache `@st.cache_resource`.
   - Form insert data baru (fitur Iris + species + timestamp otomatis).
   - Tampilan tabel koleksi MongoDB dan fitur hapus dokumen berdasarkan `_id`.
   - **Retrain Model**: Tombol retrain yang otomatis menggabungkan 150 sampel asli Scikit-Learn dengan sampel baru di MongoDB, melatih ulang Random Forest, serta menyajikan perbandingan metrik dan *Feature Importance*.

5. **💬 Komentar Publik (Guestbook Database)**:
   - Form isian komentar publik (Nama, Role/Status, Rating bintang, Pesan ulasan).
   - Data komentar tersimpan permanen di cloud MongoDB Atlas (koleksi `komentar_publik`).
   - Tampilan daftar komentar publik dengan desain kartu modern.

---

## 🛠️ Panduan Menjalankan Secara Lokal

### 1. Masuk ke Folder Proyek
```bash
cd "d:\save\Documents\TUGAS KULIAH\AI 2026 MATKUL SHARE\TUGAS STREMLIT\my-ml-app"
```

### 2. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi MongoDB Atlas (Opsional untuk Testing Lokal)
Buat file `.streamlit/secrets.toml`:
```toml
MONGODB_URI = "mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/?retryWrites=true&w=majority"
```
*(Catatan: Jika secrets belum diisi, aplikasi tetap dapat berjalan menggunakan fallback in-memory storage atau input URI langsung di sidebar).*

### 4. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 🌐 Panduan Deployment ke Streamlit Cloud & Pendaftaran UKSW

1. **Inisialisasi Git & Push ke GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit TAS Streamlit UKSW"
   git branch -M main
   git remote add origin https://github.com/<username-anda>/<nama-repo>.git
   git push -u origin main
   ```
2. **Deploy di Streamlit Community Cloud**:
   - Buka [share.streamlit.io](https://share.streamlit.io/) dan login dengan GitHub.
   - Klik **Create app** -> Pilih repository dan `app.py`.
   - Buka menu **App Settings -> Secrets**, lalu tempelkan:
     ```toml
     MONGODB_URI = "mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/?retryWrites=true&w=majority"
     ```
   - Klik **Save** dan aplikasi Anda akan live!
3. **Pendaftaran ke Web UKSW**:
   - Buka [Portal UKSW](https://mdsfsmuksw.my.id/materi-belajar/streamlit).
   - Masukkan nama dan email mahasiswa UKSW (`@student.uksw.edu`), lalu klik **Minta Token**.
   - Periksa email Anda, salin token, dan masukkan URL aplikasi Streamlit publik Anda.
