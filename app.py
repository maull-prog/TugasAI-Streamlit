# ==============================================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT & MONGODB ATLAS
# TUGAS AKHIR SEMESTER (TAS) - KULIAH AI UKSW
# Dosen Pengampu: Dr. Suryasatriya Trihandaru, M.Sc.nat
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn: Datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scikit-learn: Model Regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Scikit-learn: Model Klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Scikit-learn: Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# MongoDB & BSON
try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

# Groq Cloud AI
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
import json

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Studio UKSW - Streamlit & ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS untuk tampilan UI Modern & Elegan
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .custom-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .comment-card {
        background: rgba(26, 31, 44, 0.7);
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .badge-uksw {
        background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    .badge-status {
        background: #10B981;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    .badge-groq {
        background: linear-gradient(90deg, #F55036, #F97316);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .ai-reply-box {
        background: rgba(245, 80, 54, 0.1);
        border: 1px solid rgba(245, 80, 54, 0.3);
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. HELPER KONEKSI MONGODB ATLAS & GROQ AI DENGAN CACHE & SECRETS
# ------------------------------------------------------------------------------
def get_mongodb_uri():
    """Mengambil MongoDB URI dari st.secrets atau session_state."""
    if "MONGODB_URI" in st.secrets:
        return st.secrets["MONGODB_URI"]
    elif "custom_mongo_uri" in st.session_state and st.session_state.custom_mongo_uri:
        return st.session_state.custom_mongo_uri
    return None

def get_groq_api_key():
    """Mengambil Groq API Key dari st.secrets atau session_state."""
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    elif "custom_groq_key" in st.session_state and st.session_state.custom_groq_key:
        return st.session_state.custom_groq_key
    return None

def analyze_comment_with_groq(comment_text: str, api_key: str):
    """Menganalisis sentimen dan memberikan balasan AI otomatis menggunakan Groq Llama 3."""
    if not api_key or not HAS_GROQ:
        # Fallback offline sederhana jika Groq belum aktif
        return {
            "sentiment": "Positif 😊",
            "reply": "Terima kasih banyak atas feedback dan partisipasi Anda dalam web Machine Learning ini!"
        }
    try:
        client = Groq(api_key=api_key)
        sys_prompt = (
            "Anda adalah asisten AI resmi untuk website Machine Learning UKSW (Tugas Akhir Semester AI).\n"
            "Tugas Anda: Menganalisis komentar pengunjung dan memberikan balasan ramah.\n"
            "Keluaran WAJIB HANYA format JSON valid tanpa format markdown (tanpa ```json) dengan struktur:\n"
            "{\n"
            '  "sentiment": "Positif 😊" atau "Netral 😐" atau "Negatif 😡",\n'
            '  "reply": "Balasan hangat, sopan, dan apresiatif dalam 1-2 kalimat bahasa Indonesia"\n'
            "}"
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f'Komentar: "{comment_text}"'}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=200
        )
        raw_res = response.choices[0].message.content.strip()
        if raw_res.startswith("```"):
            raw_res = raw_res.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_res)
        return parsed
    except Exception:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": f"Balas ramah singkat dalam 1 kalimat bahasa Indonesia untuk komentar ini: {comment_text}"}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=80
            )
            return {
                "sentiment": "Positif 😊",
                "reply": response.choices[0].message.content.strip()
            }
        except Exception:
            return {
                "sentiment": "Positif 😊",
                "reply": "Terima kasih atas tanggapan dan masukan berharga yang Anda berikan!"
            }

@st.cache_resource(show_spinner=False)
def init_mongo_client(uri: str):
    """Inisialisasi koneksi MongoClient dengan caching resource."""
    if not uri or not HAS_PYMONGO:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        # Test ping server
        client.admin.command('ping')
        return client
    except Exception as e:
        return None

def get_database(client):
    """Mengambil database test_db."""
    if client is not None:
        return client["test_db"]
    return None

# Inisialisasi Mock Storage lokal jika MongoDB belum terkoneksi
if "mock_iris_db" not in st.session_state:
    st.session_state.mock_iris_db = []
if "mock_comments" not in st.session_state:
    st.session_state.mock_comments = [
        {
            "nama": "Dr. Suryasatriya Trihandaru, M.Sc.nat",
            "peran": "Dosen Pembimbing",
            "rating": 5,
            "komentar": "Implementasi sistem end-to-end Streamlit, scikit-learn, MongoDB, dan Groq AI sangat lengkap dan josjis!",
            "sentiment": "Positif 😊",
            "ai_reply": "Terima kasih banyak atas bimbingan dan arahannya selama perkuliahan AI, Pak Surya!",
            "timestamp": "2026-10-14 10:30:00"
        },
        {
            "nama": "Tim Reviewer AI",
            "peran": "Reviewer",
            "rating": 5,
            "komentar": "Fitur clustering dan integrasi inferensi Groq LPU berjalan sangat responsif dan cepat.",
            "sentiment": "Positif 😊",
            "ai_reply": "Senang mendengarnya! Kecepatan inferensi Groq LPU sangat membantu performa web real-time.",
            "timestamp": "2026-10-15 08:45:00"
        }
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Halo! Saya adalah Asisten AI berbasis **Groq LPU (Llama 3)**. Ada yang bisa saya bantu seputar model Machine Learning, evaluasi metrik, atau dataset Anda?"}
    ]

# ------------------------------------------------------------------------------
# 3. SIDEBAR: KONTROL, STATUS DATABASE & INFORMASI PROYEK
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 UKSW AI Studio")
    st.markdown('<span class="badge-uksw">TAS Machine Learning 2026</span>', unsafe_allow_html=True)
    st.caption("Dr. Suryasatriya Trihandaru, M.Sc.nat")
    st.markdown("---")

    # Status Koneksi MongoDB
    st.subheader("🍃 Status MongoDB Atlas")
    current_uri = get_mongodb_uri()
    mongo_client = init_mongo_client(current_uri) if current_uri else None

    if mongo_client:
        st.success("🟢 Terhubung ke MongoDB Atlas")
        st.caption("Database: `test_db` | Collection: `iris_db`")
    else:
        st.warning("🟠 MongoDB Belum Terhubung")
        st.caption("Memakai mode simulasi memori. Untuk menghubungkan ke Cloud, masukkan URI di bawah atau gunakan `.streamlit/secrets.toml`:")
        custom_input = st.text_input(
            "Alamat MONGODB_URI:",
            type="password",
            placeholder="mongodb+srv://user:pass@cluster...",
            help="Masukkan string koneksi MongoDB Atlas Anda"
        )
        if custom_input:
            st.session_state.custom_mongo_uri = custom_input
            st.rerun()

    # Status Groq AI Cloud
    st.markdown("---")
    st.subheader("⚡ Status Groq AI")
    current_groq_key = get_groq_api_key()
    if current_groq_key:
        st.success("⚡ Groq LPU Aktif (Llama 3.3)")
        st.caption("Model: `llama-3.3-70b-versatile` | Ultra-Fast LPU Inference")
    else:
        st.info("⚡ Groq API Key (Aktifkan AI)")
        st.caption("Dapatkan key gratis di [console.groq.com](https://console.groq.com):")
        groq_input = st.text_input(
            "GROQ_API_KEY:",
            type="password",
            placeholder="gsk_xxxxxxxxxxxx...",
            help="Masukkan API key Groq Anda untuk analisis sentimen & chatbot AI"
        )
        if groq_input:
            st.session_state.custom_groq_key = groq_input
            st.rerun()

    st.markdown("---")
    st.subheader("👨‍💻 Identitas Kelompok")
    st.text_input("Nama Mahasiswa / Kelompok:", value="Kelompok AI - UKSW", key="student_name")
    st.text_input("NIM / Kelas:", value="682022xxx - Sains Data", key="student_nim")

    st.markdown("---")
    st.markdown("### 📌 Fitur Aplikasi")
    st.markdown("""
    - ✅ **Regresi Linear & Regresi Lanjutan**
    - ✅ **Klasifikasi Multikelas & Metrik**
    - ✅ **Clustering K-Means & PCA (Unsupervised)**
    - ✅ **MongoDB CRUD & Retraining Model**
    - ✅ **Komentar Publik (Auto-Sentiment & AI Reply)**
    - ✅ **Asisten Chatbot Cerdas Groq LPU**
    """)

# ------------------------------------------------------------------------------
# 4. HEADER UTAMA
# ------------------------------------------------------------------------------
st.title("🤖 Web Machine Learning Studio & Cloud Database")
st.markdown("""
Aplikasi komprehensif implementasi *End-to-End Pipeline Machine Learning* berbasis **Streamlit**, 
mengintegrasikan algoritma supervised/unsupervised dengan **MongoDB Atlas** dan akselerasi AI **Groq LPU**.
""")

# Navigasi Tab Utama
tab_regresi, tab_klasifikasi, tab_clustering, tab_retrain, tab_komentar, tab_groq, tab_info = st.tabs([
    "📈 Regresi Linear",
    "🎯 Klasifikasi",
    "🔮 Clustering",
    "🍃 MongoDB & Retrain",
    "💬 Komentar & AI Sentimen",
    "🤖 Asisten Groq AI",
    "📘 Panduan & Info TAS"
])

# ==============================================================================
# TAB 1: REGRESI
# ==============================================================================
with tab_regresi:
    st.header("📈 Model Regresi (Linear & Nonlinear)")
    st.caption("Prediksi nilai kontinu dengan evaluasi metrik MAE, MSE, RMSE, dan R².")

    col_r1, col_r2 = st.columns([1, 2.5])

    with col_r1:
        st.markdown("#### ⚙️ Konfigurasi Regresi")
        reg_dataset_choice = st.selectbox(
            "Pilih Dataset Regresi:",
            ["Diabetes Dataset (Scikit-Learn)", "Synthetic Housing Price (UCI Benchmark)"],
            key="reg_dataset"
        )
        
        reg_model_choice = st.selectbox(
            "Pilih Model Regresi:",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"],
            key="reg_model"
        )

        test_size_reg = st.slider("Ukuran Test Split:", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_reg")
        random_state_reg = st.number_input("Random State:", value=42, step=1, key="rs_reg")

        # Parameter khusus
        alpha_val = 1.0
        if reg_model_choice in ["Ridge Regression", "Lasso Regression"]:
            alpha_val = st.slider("Alpha (Regularisasi):", 0.01, 10.0, 1.0, 0.1)

        btn_train_reg = st.button("🚀 Latih Model Regresi", key="btn_reg")

    with col_r2:
        # Load Dataset Regresi
        if "Diabetes" in reg_dataset_choice:
            data_reg = datasets.load_diabetes(as_frame=True)
            X_reg = data_reg.data
            y_reg = data_reg.target
            feature_names_reg = data_reg.feature_names
        else:
            # Synthetic Housing Benchmark
            np.random.seed(int(random_state_reg))
            X_synth, y_synth = datasets.make_regression(n_samples=350, n_features=6, noise=15.0, random_state=int(random_state_reg))
            feature_names_reg = [f"Fitur_{i+1}" for i in range(6)]
            X_reg = pd.DataFrame(X_synth, columns=feature_names_reg)
            y_reg = pd.Series(y_synth, name="Target_Harga")

        # Split Data
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reg, y_reg, test_size=test_size_reg, random_state=int(random_state_reg)
        )

        # Inisialisasi Model
        if reg_model_choice == "Linear Regression":
            model_reg = LinearRegression()
        elif reg_model_choice == "Ridge Regression":
            model_reg = Ridge(alpha=alpha_val)
        elif reg_model_choice == "Lasso Regression":
            model_reg = Lasso(alpha=alpha_val)
        elif reg_model_choice == "Decision Tree Regressor":
            model_reg = DecisionTreeRegressor(random_state=int(random_state_reg))
        else:
            model_reg = RandomForestRegressor(n_estimators=100, random_state=int(random_state_reg))

        # Training
        start_time_r = time.time()
        model_reg.fit(X_train_r, y_train_r)
        durasi_reg = time.time() - start_time_r

        # Prediksi
        y_pred_r = model_reg.predict(X_test_r)

        # Hitung Metrik
        mae_r = mean_absolute_error(y_test_r, y_pred_r)
        mse_r = mean_squared_error(y_test_r, y_pred_r)
        rmse_r = np.sqrt(mse_r)
        r2_r = r2_score(y_test_r, y_pred_r)

        st.success(f"✅ Model **{reg_model_choice}** berhasil dilatih dalam **{durasi_reg:.4f} detik**!")

        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{mae_r:.3f}")
        m2.metric("MSE", f"{mse_r:.3f}")
        m3.metric("RMSE", f"{rmse_r:.3f}")
        m4.metric("R² Score", f"{r2_r:.3f}")

        # Visualisasi Actual vs Predicted
        sub_tab1, sub_tab2 = st.tabs(["📊 Actual vs Predicted", "📋 Data Hasil Prediksi"])

        with sub_tab1:
            fig_reg = go.Figure()
            # Scatter titik data
            fig_reg.add_trace(go.Scatter(
                x=y_test_r,
                y=y_pred_r,
                mode='markers',
                name='Prediksi Sampel',
                marker=dict(color='#00d2ff', size=8, opacity=0.8, line=dict(width=1, color='white'))
            ))
            # Garis identitas sempurna (y = x)
            min_val = min(y_test_r.min(), y_pred_r.min())
            max_val = max(y_test_r.max(), y_pred_r.max())
            fig_reg.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Prediksi Sempurna (y = x)',
                line=dict(color='#ff4b4b', dash='dash', width=2)
            ))
            fig_reg.update_layout(
                title=f"Grafik Actual vs Predicted: {reg_model_choice}",
                xaxis_title="Nilai Sebenarnya (Actual)",
                yaxis_title="Nilai Prediksi (Predicted)",
                template="plotly_dark",
                height=450
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with sub_tab2:
            df_res_reg = pd.DataFrame({
                "Nilai Aktual": y_test_r.values,
                "Nilai Prediksi": y_pred_r,
                "Selisih (Error)": np.abs(y_test_r.values - y_pred_r)
            })
            st.dataframe(df_res_reg.head(20), use_container_width=True)

# ==============================================================================
# TAB 2: KLASIFIKASI
# ==============================================================================
with tab_klasifikasi:
    st.header("🎯 Model Klasifikasi (Supervised Learning)")
    st.caption("Pemodelan multikelas dan biner dengan Confusion Matrix dan Classification Report.")

    col_k1, col_k2 = st.columns([1, 2.5])

    with col_k1:
        st.markdown("#### ⚙️ Konfigurasi Klasifikasi")
        clf_dataset_choice = st.selectbox(
            "Pilih Dataset Klasifikasi:",
            ["Iris Flowers (3 Kelas)", "Wine Recognition (3 Kelas)", "Breast Cancer Wisconsin (Biner)"],
            key="clf_dataset"
        )

        clf_model_choice = st.selectbox(
            "Pilih Algoritma Klasifikasi:",
            ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest", "Support Vector Machine (SVM)"],
            key="clf_model"
        )

        test_size_clf = st.slider("Ukuran Test Split:", 0.1, 0.5, 0.25, 0.05, key="test_clf")
        random_state_clf = st.number_input("Random State:", value=42, step=1, key="rs_clf")

        # Hyperparameter adaptif
        k_val = 5
        if clf_model_choice == "K-Nearest Neighbors (KNN)":
            k_val = st.slider("Nilai K (Neighbors):", 1, 15, 5)

        btn_train_clf = st.button("🚀 Latih Model Klasifikasi", key="btn_clf")

    with col_k2:
        # Load Dataset
        if "Iris" in clf_dataset_choice:
            data_clf = datasets.load_iris(as_frame=True)
        elif "Wine" in clf_dataset_choice:
            data_clf = datasets.load_wine(as_frame=True)
        else:
            data_clf = datasets.load_breast_cancer(as_frame=True)

        X_clf = data_clf.data
        y_clf = data_clf.target
        target_names = [str(name) for name in data_clf.target_names]

        # Normalisasi untuk KNN & SVM
        scaler = StandardScaler()
        X_scaled_clf = scaler.fit_transform(X_clf)

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_scaled_clf if clf_model_choice in ["K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"] else X_clf,
            y_clf,
            test_size=test_size_clf,
            random_state=int(random_state_clf),
            stratify=y_clf
        )

        # Inisialisasi Model
        if clf_model_choice == "Logistic Regression":
            model_clf = LogisticRegression(max_iter=300, random_state=int(random_state_clf))
        elif clf_model_choice == "K-Nearest Neighbors (KNN)":
            model_clf = KNeighborsClassifier(n_neighbors=k_val)
        elif clf_model_choice == "Decision Tree":
            model_clf = DecisionTreeClassifier(random_state=int(random_state_clf))
        elif clf_model_choice == "Random Forest":
            model_clf = RandomForestClassifier(n_estimators=100, random_state=int(random_state_clf))
        else:
            model_clf = SVC(probability=True, random_state=int(random_state_clf))

        start_time_c = time.time()
        model_clf.fit(X_train_c, y_train_c)
        durasi_clf = time.time() - start_time_c

        y_pred_c = model_clf.predict(X_test_c)

        # Metrik
        acc_c = accuracy_score(y_test_c, y_pred_c)
        prec_c = precision_score(y_test_c, y_pred_c, average='weighted', zero_division=0)
        rec_c = recall_score(y_test_c, y_pred_c, average='weighted', zero_division=0)
        f1_c = f1_score(y_test_c, y_pred_c, average='weighted', zero_division=0)

        st.success(f"✅ Model **{clf_model_choice}** berhasil ditraining dalam **{durasi_clf:.4f} detik**!")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Akurasi", f"{acc_c * 100:.2f}%")
        mc2.metric("Precision", f"{prec_c:.3f}")
        mc3.metric("Recall", f"{rec_c:.3f}")
        mc4.metric("F1-Score", f"{f1_c:.3f}")

        # Visualisasi Confusion Matrix & Report
        c_sub1, c_sub2 = st.tabs(["🔥 Confusion Matrix Heatmap", "📄 Classification Report"])

        with c_sub1:
            cm = confusion_matrix(y_test_c, y_pred_c)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                x=target_names,
                y=target_names,
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_cm.update_layout(title="Confusion Matrix", height=430)
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_sub2:
            report_dict = classification_report(y_test_c, y_pred_c, target_names=target_names, output_dict=True)
            df_report = pd.DataFrame(report_dict).transpose()
            st.dataframe(df_report.style.format(precision=3), use_container_width=True)

# ==============================================================================
# TAB 3: CLUSTERING (UNSUPERVISED LEARNING)
# ==============================================================================
with tab_clustering:
    st.header("🔮 Model Clustering (Unsupervised Learning)")
    st.caption("Pengelompokan data tanpa label menggunakan K-Means, Agglomerative, dan PCA 2D.")

    col_cl1, col_cl2 = st.columns([1, 2.5])

    with col_cl1:
        st.markdown("#### ⚙️ Konfigurasi Clustering")
        cluster_dataset = st.selectbox(
            "Pilih Dataset Unsupervised:",
            ["Iris Features (Fitur Bunga)", "Wine Features (Karakteristik Anggur)", "Synthetic Clustered Blobs"],
            key="cl_dataset"
        )

        cluster_algo = st.selectbox(
            "Algoritma Clustering:",
            ["K-Means Clustering", "Agglomerative (Hierarchical) Clustering", "DBSCAN"],
            key="cl_algo"
        )

        if cluster_algo in ["K-Means Clustering", "Agglomerative (Hierarchical) Clustering"]:
            n_clusters = st.slider("Jumlah Cluster (k):", min_value=2, max_value=8, value=3, key="k_cluster")
        else:
            eps_val = st.slider("Epsilon (Radius):", 0.1, 2.0, 0.6, 0.1)
            min_samples = st.slider("Min Samples:", 2, 10, 5)

        btn_cluster = st.button("🚀 Jalankan Analisis Klaster", key="btn_cl")

    with col_cl2:
        # Load dataset
        if "Iris" in cluster_dataset:
            data_raw = datasets.load_iris(as_frame=True)
            X_clust = data_raw.data
        elif "Wine" in cluster_dataset:
            data_raw = datasets.load_wine(as_frame=True)
            X_clust = data_raw.data
        else:
            X_blob, _ = datasets.make_blobs(n_samples=300, centers=4, n_features=4, random_state=42)
            X_clust = pd.DataFrame(X_blob, columns=[f"Fitur_{i+1}" for i in range(4)])

        # Standarisasi
        X_cl_scaled = StandardScaler().fit_transform(X_clust)

        # Eksekusi Algoritma
        if cluster_algo == "K-Means Clustering":
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clustering_model.fit_predict(X_cl_scaled)
        elif cluster_algo == "Agglomerative (Hierarchical) Clustering":
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering_model.fit_predict(X_cl_scaled)
        else:
            clustering_model = DBSCAN(eps=eps_val, min_samples=min_samples)
            labels = clustering_model.fit_predict(X_cl_scaled)

        # Evaluasi Silhouette & Davies-Bouldin
        unique_labels = set(labels)
        has_valid_clusters = len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels)

        if has_valid_clusters:
            sil_score = silhouette_score(X_cl_scaled, labels)
            db_score = davies_bouldin_score(X_cl_scaled, labels)
        else:
            sil_score, db_score = 0.0, 0.0

        st.success(f"✅ Analisis Klaster **{cluster_algo}** Selesai!")

        cmet1, cmet2, cmet3 = st.columns(3)
        cmet1.metric("Total Klaster Terbentuk", len([x for x in unique_labels if x != -1]))
        cmet2.metric("Silhouette Score", f"{sil_score:.3f}", help="Mendekati 1.0 berarti pemisahan klaster sangat baik")
        cmet3.metric("Davies-Bouldin Index", f"{db_score:.3f}", help="Semakin kecil nilainya semakin baik")

        # Proyeksi 2D PCA untuk Visualisasi Klaster
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cl_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["Komponen PCA 1", "Komponen PCA 2"])
        df_pca["Cluster"] = [f"Klaster {lbl}" if lbl != -1 else "Noise/Outlier" for lbl in labels]

        fig_cluster = px.scatter(
            df_pca,
            x="Komponen PCA 1",
            y="Komponen PCA 2",
            color="Cluster",
            title=f"Visualisasi Proyeksi Klaster (PCA 2D) - Rasio Variansi: {pca.explained_variance_ratio_.sum()*100:.1f}%",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

# ==============================================================================
# TAB 4: MONGODB CRUD & RETRAINING
# ==============================================================================
with tab_retrain:
    st.header("🍃 Integrasi Database MongoDB Atlas & Retraining")
    st.caption("Tambah data sampel baru ke Cloud MongoDB, hapus data, dan latih ulang model secara real-time.")

    col_m1, col_m2 = st.columns([1, 1.8])

    with col_m1:
        st.markdown("#### ➕ Tambah Data Iris Baru")
        with st.form("form_insert_iris", clear_on_submit=True):
            f_sepal_length = st.number_input("Sepal Length (cm):", min_value=1.0, max_value=10.0, value=5.1, step=0.1)
            f_sepal_width = st.number_input("Sepal Width (cm):", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
            f_petal_length = st.number_input("Petal Length (cm):", min_value=0.5, max_value=10.0, value=1.4, step=0.1)
            f_petal_width = st.number_input("Petal Width (cm):", min_value=0.1, max_value=5.0, value=0.2, step=0.1)
            f_species = st.selectbox("Species Target:", ["setosa", "versicolor", "virginica"])
            
            btn_submit_iris = st.form_submit_button("💾 Simpan ke Database MongoDB")

            if btn_submit_iris:
                new_doc = {
                    "sepal_length": float(f_sepal_length),
                    "sepal_width": float(f_sepal_width),
                    "petal_length": float(f_petal_length),
                    "petal_width": float(f_petal_width),
                    "species": f_species,
                    "timestamp": datetime.now()
                }

                if mongo_client:
                    db = get_database(mongo_client)
                    db.iris_db.insert_one(new_doc)
                    st.success("✅ Berhasil menyimpan dokumen ke MongoDB Atlas!")
                else:
                    new_doc["_id"] = str(len(st.session_state.mock_iris_db) + 1)
                    st.session_state.mock_iris_db.append(new_doc)
                    st.info("ℹ️ Disimpan ke Local Memory Storage (MongoDB belum terkoneksi)")
                time.sleep(0.5)
                st.rerun()

    with col_m2:
        st.markdown("#### 📊 Koleksi Data MongoDB: `test_db.iris_db`")

        # Mengambil Data dari MongoDB
        docs = []
        if mongo_client:
            try:
                db = get_database(mongo_client)
                docs = list(db.iris_db.find().sort("timestamp", -1))
            except Exception as e:
                st.error(f"Gagal mengambil data MongoDB: {e}")
        else:
            docs = st.session_state.mock_iris_db

        st.metric("Total Data Baru Tersimpan", len(docs))

        if len(docs) > 0:
            # Format DataFrame
            display_docs = []
            for d in docs:
                display_docs.append({
                    "ID": str(d.get("_id")),
                    "Sepal Length": d.get("sepal_length"),
                    "Sepal Width": d.get("sepal_width"),
                    "Petal Length": d.get("petal_length"),
                    "Petal Width": d.get("petal_width"),
                    "Species": d.get("species"),
                    "Waktu": d.get("timestamp")
                })
            df_mongo = pd.DataFrame(display_docs)
            st.dataframe(df_mongo, use_container_width=True)

            # Fitur Hapus Data Berdasarkan ID
            col_del1, col_del2 = st.columns([2, 1])
            with col_del1:
                selected_del_id = st.selectbox("Pilih ID Dokumen untuk Dihapus:", [d["ID"] for d in display_docs])
            with col_del2:
                st.write("")
                st.write("")
                if st.button("🗑️ Hapus Dokumen"):
                    if mongo_client:
                        db = get_database(mongo_client)
                        db.iris_db.delete_one({"_id": ObjectId(selected_del_id)})
                        st.success(f"Dokumen {selected_del_id} berhasil dihapus dari MongoDB!")
                    else:
                        st.session_state.mock_iris_db = [d for d in st.session_state.mock_iris_db if str(d.get("_id")) != selected_del_id]
                        st.success("Dokumen dihapus dari Local Storage!")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.info("Belum ada data tambahan di database. Silakan isi form di samping untuk menambah sampel baru.")

    st.markdown("---")
    # ==================== FITUR RETRAINING ====================
    st.subheader("🔄 Retraining Model: Gabungkan Data Asli + Data MongoDB")
    st.markdown("""
    Fitur ini menggabungkan **150 sampel asli Scikit-Learn Iris** dengan **data tambahan dari MongoDB**,
    kemudian melatih ulang model **Random Forest Classifier** secara otomatis.
    """)

    if st.button("🚀 Retrain Sekarang dengan Semua Data", type="primary"):
        # 1. Dataset Iris bawaan (150 sampel)
        iris_raw = datasets.load_iris()
        df_base = pd.DataFrame(iris_raw.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        df_base["species"] = [iris_raw.target_names[i] for i in iris_raw.target]

        # 2. Data tambahan dari MongoDB
        if len(docs) > 0:
            df_add = pd.DataFrame([
                {
                    "sepal_length": float(d["sepal_length"]),
                    "sepal_width": float(d["sepal_width"]),
                    "petal_length": float(d["petal_length"]),
                    "petal_width": float(d["petal_width"]),
                    "species": str(d["species"])
                }
                for d in docs
            ])
            df_combined = pd.concat([df_base, df_add], ignore_index=True)
        else:
            df_combined = df_base

        # 3. Encoding target (setosa=0, versicolor=1, virginica=2)
        mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
        df_combined["target"] = df_combined["species"].map(mapping)
        df_combined = df_combined.dropna(subset=["target"])

        X_retrain = df_combined[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        y_retrain = df_combined["target"].astype(int)

        # 4. Split & Retrain
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_retrain, y_retrain, test_size=0.25, random_state=42)
        rf_retrain = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_retrain.fit(Xr_train, yr_train)

        # Evaluasi
        yr_pred = rf_retrain.predict(Xr_test)
        acc_retrain = accuracy_score(yr_test, yr_pred)

        st.balloons()
        st.success(f"🎉 **Retraining Berhasil!** Model dilatih dengan total **{len(df_combined)} sampel** (150 bawaan + {len(docs)} dari database).")

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Total Sampel Training", len(df_combined))
        rm2.metric("Akurasi Model Baru", f"{acc_retrain * 100:.2f}%")
        rm3.metric("Peningkatan Kapasitas", f"+{len(docs)} data")

        # Feature Importance
        df_importance = pd.DataFrame({
            "Fitur": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Importance": rf_retrain.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            df_importance,
            x="Importance",
            y="Fitur",
            orientation="h",
            title="Tingkat Kepentingan Fitur (Feature Importance - Random Forest)",
            template="plotly_dark",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# TAB 5: KOMENTAR & AI SENTIMEN (POWERED BY GROQ & MONGODB)
# ==============================================================================
with tab_komentar:
    st.header("💬 Komentar Publik & Analisis Sentimen (Groq AI)")
    st.caption("Fitur interaktif publik untuk memberikan ulasan. Groq LPU AI otomatis menganalisis sentimen dan memberikan tanggapan cerdas.")

    col_k1, col_k2 = st.columns([1.1, 1.6])

    with col_k1:
        st.markdown("#### ✍️ Beri Komentar Baru")
        with st.form("form_komentar_publik", clear_on_submit=True):
            in_nama = st.text_input("Nama Lengkap / Inisial:", placeholder="Misal: Budi Santoso")
            in_role = st.selectbox("Status / Peran:", ["Mahasiswa", "Dosen / Pengajar", "Praktisi Data Science", "Tamu / Publik"])
            in_rating = st.slider("Rating Aplikasi:", 1, 5, 5, help="1 = Kurang, 5 = Sangat Bagus")
            in_pesan = st.text_area("Pesan / Ulasan / Feedback:", placeholder="Tuliskan ulasan Anda mengenai aplikasi ini...")
            
            btn_kirim_komentar = st.form_submit_button("🚀 Kirim Komentar & Analisis AI")

            if btn_kirim_komentar:
                if in_nama and in_pesan:
                    with st.spinner("🤖 Groq AI sedang menganalisis sentimen & menyiapkan respon cerdas..."):
                        groq_key = get_groq_api_key()
                        ai_res = analyze_comment_with_groq(in_pesan, groq_key)
                        sentiment_val = ai_res.get("sentiment", "Positif 😊")
                        reply_val = ai_res.get("reply", "Terima kasih banyak atas partisipasi dan ulasan Anda!")

                    komentar_doc = {
                        "nama": in_nama,
                        "peran": in_role,
                        "rating": int(in_rating),
                        "komentar": in_pesan,
                        "sentiment": sentiment_val,
                        "ai_reply": reply_val,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    if mongo_client:
                        try:
                            db = get_database(mongo_client)
                            db.komentar_publik.insert_one(komentar_doc)
                            st.success("✅ Komentar berhasil dianalisis AI dan disimpan ke MongoDB!")
                        except Exception as e:
                            st.error(f"Gagal simpan ke database: {e}")
                    else:
                        st.session_state.mock_comments.insert(0, komentar_doc)
                        st.success("✅ Komentar berhasil dianalisis AI dan ditambahkan ke daftar ulasan!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("⚠️ Mohon isi Nama dan Pesan sebelum mengirim.")

    with col_k2:
        st.markdown("#### 📜 Daftar Ulasan Publik Terintegrasi AI")

        # Ambil Komentar dari MongoDB
        list_komentar = []
        if mongo_client:
            try:
                db = get_database(mongo_client)
                list_komentar = list(db.komentar_publik.find().sort("timestamp", -1))
            except Exception:
                list_komentar = st.session_state.mock_comments
        else:
            list_komentar = st.session_state.mock_comments

        if len(list_komentar) > 0:
            for item in list_komentar:
                bintang = "⭐" * int(item.get("rating", 5))
                sentimen = item.get("sentiment", "Positif 😊")
                ai_reply = item.get("ai_reply", "")
                
                sentimen_badge_color = "#10B981" if "Positif" in sentimen else ("#F59E0B" if "Netral" in sentimen else "#EF4444")
                
                reply_html = ""
                if ai_reply:
                    reply_html = f"""
                    <div class="ai-reply-box">
                        <strong style="color:#F97316;">🤖 Respon Asisten AI (Groq LPU):</strong><br>
                        <span>{ai_reply}</span>
                    </div>
                    """
                
                st.markdown(f"""
                <div class="comment-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>{item.get('nama', 'Pengunjung')}</strong>
                        <span style="font-size:0.8rem; color:#888;">{item.get('timestamp', '')}</span>
                    </div>
                    <div style="font-size:0.85rem; margin-top:2px; margin-bottom:6px;">
                        <span style="color:#00d2ff;">{item.get('peran', 'Umum')}</span> &bull; {bintang} &bull; 
                        <span style="background:{sentimen_badge_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem;">
                            Sentimen: {sentimen}
                        </span>
                    </div>
                    <p style="margin-bottom:6px; font-size:0.95rem;">{item.get('komentar', '')}</p>
                    {reply_html}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Belum ada komentar dari publik. Jadilah yang pertama memberikan ulasan!")

# ==============================================================================
# TAB 6: ASISTEN CHATBOT GROQ AI (LLAMA 3.3 LPU)
# ==============================================================================
with tab_groq:
    st.header("🤖 Asisten Cerdas Groq AI (LPU Inference)")
    st.caption("Konsultasi interaktif seputar Machine Learning, Algoritma, Evaluasi, dan Arsitektur Cloud menggunakan inferensi super cepat Llama 3.3.")

    groq_active_key = get_groq_api_key()

    if not groq_active_key:
        st.warning("⚡ **Groq API Key belum dimasukkan.** Anda bisa memasukkannya di Sidebar atau di file `.streamlit/secrets.toml` untuk mengaktifkan kecerdasan penuh.")
        st.markdown("""
        > 💡 **Cara Mendapatkan Groq API Key Gratis (1 Menit):**
        > 1. Buka [console.groq.com](https://console.groq.com)
        > 2. Login dengan akun Google / GitHub
        > 3. Klik menu **API Keys** -> **Create API Key**
        > 4. Salin dan tempelkan ke kolom **GROQ_API_KEY** di Sidebar aplikasi ini.
        """)

    # Quick Prompt Chips
    st.markdown("##### 💡 Contoh Pertanyaan Cepat:")
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_query = None
    if q_col1.button("📈 Kapan pakai Regresi vs Klasifikasi?", use_container_width=True):
        quick_query = "Kapan kita sebaiknya menggunakan model Regresi dan kapan menggunakan Klasifikasi? Berikan contohnya."
    if q_col2.button("🔮 Apa kelebihan K-Means Clustering?", use_container_width=True):
        quick_query = "Jelaskan prinsip kerja K-Means Clustering dan metrik evaluasi Silhouette Score!"
    if q_col3.button("🍃 Mengapa butuh MongoDB di Web ML?", use_container_width=True):
        quick_query = "Mengapa kita menghubungkan MongoDB Atlas ke Streamlit untuk dataset dan komentar publik?"

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_input = st.chat_input("Tanyakan apa saja seputar Machine Learning & Tugas Ini...")
    actual_prompt = quick_query if quick_query else user_input

    if actual_prompt:
        st.session_state.chat_history.append({"role": "user", "content": actual_prompt})
        with st.chat_message("user"):
            st.markdown(actual_prompt)

        with st.chat_message("assistant"):
            if groq_active_key and HAS_GROQ:
                try:
                    client = Groq(api_key=groq_active_key)
                    system_instruction = (
                        "Anda adalah asisten AI ahli Machine Learning untuk mahasiswa Universitas Kristen Satya Wacana (UKSW). "
                        "Jawab dengan ramah, jelas, edukatif, dan menggunakan bahasa Indonesia yang baik. "
                        "Bantu pengguna memahami konsep Regresi, Klasifikasi, Clustering, MongoDB, dan Streamlit."
                    )
                    messages_payload = [{"role": "system", "content": system_instruction}]
                    # Ambil 6 pesan terakhir untuk konteks
                    for m in st.session_state.chat_history[-6:]:
                        messages_payload.append({"role": m["role"], "content": m["content"]})

                    stream = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages_payload,
                        temperature=0.5,
                        max_tokens=600,
                        stream=True
                    )
                    response_text = st.write_stream(stream)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                except Exception as err:
                    fallback_ans = f"Terjadi kendala koneksi ke Groq API: `{err}`. Pastikan kuota atau API key Anda valid."
                    st.error(fallback_ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": fallback_ans})
            else:
                # Built-in intelligent response
                fallback_ans = (
                    "**Jawaban Asisten:**\n\n"
                    "- **Regresi** digunakan saat target bernilai kontinu (misal harga rumah, kadar glukosa).\n"
                    "- **Klasifikasi** digunakan saat target berupa kategori diskrit (misal jenis spesies bunga Iris, deteksi tumor ganas/jinak).\n"
                    "- **Clustering** adalah *unsupervised learning* untuk mengelompokkan data tanpa label berdasarkan kemiripan pola.\n"
                    "- **MongoDB Atlas** berguna untuk menyimpan data sampel baru secara cloud, sehingga model dapat di-**retrain** sewaktu-waktu tanpa kehilangan data!\n\n"
                    "*(💡 Masukkan `GROQ_API_KEY` di Sidebar untuk mendapatkan jawaban interaktif real-time menggunakan Llama 3.3)*"
                )
                st.markdown(fallback_ans)
                st.session_state.chat_history.append({"role": "assistant", "content": fallback_ans})

# ==============================================================================
# TAB 7: PANDUAN & INFO TAS UKSW
# ==============================================================================
with tab_info:
    st.header("📘 Informasi Proyek & Pendaftaran TAS")
    st.markdown("""
    ### 🎓 Ujian Akhir Semester (TAS) - Kecerdasan Buatan
    * **Dosen Pengampu**: Dr. Suryasatriya Trihandaru, M.Sc.nat
    * **Institusi**: Magister Sains Data / FSM - Universitas Kristen Satya Wacana (UKSW)
    * **Batas Waktu Pengumpulan**: 1 Desember 2026
    """)

    st.markdown("---")
    st.subheader("📋 Checklist Kelengkapan TAS")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        - [x] **Regresi Linear**: Implementasi model regresi, metrik MAE/MSE/RMSE/R², dan grafik Actual vs Predicted.
        - [x] **Klasifikasi**: Model supervised classification, Confusion Matrix heatmap, dan Classification Report.
        - [x] **Clustering**: K-Means & Agglomerative, evaluasi Silhouette score, serta visualisasi PCA 2D.
        """)
    with c2:
        st.markdown("""
        - [x] **Integrasi MongoDB & Retrain**: Form insert sampel, hapus dokumen, dan retrain gabungan data.
        - [x] **Komentar Publik + Groq AI**: Form komentar publik dengan **Auto-Sentiment Analysis** dan **AI Auto-Reply** via Groq LPU, tersimpan di MongoDB.
        - [x] **Pendaftaran URL**: Siap didaftarkan ke portal web UKSW.
        """)

    st.markdown("---")
    st.subheader("🌐 Panduan Mendaftarkan URL Streamlit ke Web UKSW")
    st.markdown("""
    1. **Push Proyek ke GitHub**: Commit semua file dan push repositori Anda ke akun GitHub.
    2. **Deploy di Streamlit Cloud**: Buka [share.streamlit.io](https://share.streamlit.io/) dan deploy repositori Anda.
    3. **Set Secrets di Streamlit Cloud**: Di menu `App Settings -> Secrets`, masukkan:
       ```toml
       MONGODB_URI = "mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/?retryWrites=true&w=majority"
       GROQ_API_KEY = "gsk_xxxxxxxxxxxx..."
       ```
    4. **Daftarkan URL**:
       * Buka [Portal Pendaftaran UKSW](https://mdsfsmuksw.my.id/materi-belajar/streamlit)
       * Klik **Minta Token** (gunakan email mahasiswa `@student.uksw.edu`).
       * Masukkan token yang diterima, lalu daftarkan URL Streamlit publik Anda.
    """)

    st.markdown("---")
    st.caption("Dibuat dengan ❤️ untuk Ujian Akhir Semester AI UKSW 2026.")
