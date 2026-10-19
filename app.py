"""
Dashboard Machine Learning Interaktif
Regresi Linear, Klasifikasi, Clustering, Retraining Real-Time dengan MongoDB Atlas & Akselerasi AI Groq LPU
"""

import os
import pickle
import json
from datetime import datetime, timedelta, timezone
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Scikit-learn
from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_iris,
    load_wine,
    load_breast_cancer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# MongoDB & Groq
try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ----------------------------------------------------------------------------
# 1. Konfigurasi Halaman & Tema Hedon / Pro Cyberpunk
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Machine Learning",
    page_icon="🧠🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - Ultra Luxury Cyberpunk / Pro Developer "Hedon" Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #06080e !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(121, 40, 202, 0.22) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(0, 245, 255, 0.18) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(255, 0, 128, 0.15) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
    }

    /* Top Developer Badge */
    .dev-badge-top {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(90deg, rgba(255, 0, 128, 0.2), rgba(0, 245, 255, 0.2));
        border: 1px solid rgba(0, 245, 255, 0.4);
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #00F5FF;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.3);
        margin-bottom: 12px;
    }

    /* Luxury Hologram Title */
    .hero-title-hedon {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.9rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        line-height: 1.15 !important;
        background: linear-gradient(135deg, #00F5FF 0%, #A855F7 50%, #FF007A 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        filter: drop-shadow(0 0 30px rgba(168, 85, 247, 0.4)) !important;
        margin-bottom: 6px !important;
    }
    .hero-subtitle-hedon {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
        max-width: 900px !important;
        margin-bottom: 1.8rem !important;
    }

    /* "Hedon" Luxury Glassmorphism Cards */
    .hedon-card {
        background: rgba(13, 17, 28, 0.75) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-radius: 20px !important;
        padding: 26px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.7) !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        height: 100% !important;
    }
    .hedon-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #FF007A, #A855F7, #00F5FF);
    }
    .hedon-card:hover {
        transform: translateY(-8px) scale(1.02) !important;
        border-color: rgba(0, 245, 255, 0.5) !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.9), 0 0 30px rgba(0, 245, 255, 0.3) !important;
    }

    .card-badge {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .badge-reg { background: rgba(0, 245, 255, 0.15); color: #00F5FF; border: 1px solid rgba(0, 245, 255, 0.4); box-shadow: 0 0 10px rgba(0, 245, 255, 0.2); }
    .badge-clf { background: rgba(255, 0, 122, 0.15); color: #FF007A; border: 1px solid rgba(255, 0, 122, 0.4); box-shadow: 0 0 10px rgba(255, 0, 122, 0.2); }
    .badge-clu { background: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.4); box-shadow: 0 0 10px rgba(168, 85, 247, 0.2); }

    .hedon-title {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
    }
    .hedon-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    /* Pro Buttons (Hedon Neon Glow) */
    .stButton > button {
        background: linear-gradient(135deg, #7928CA 0%, #00F5FF 100%) !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.6rem !important;
        box-shadow: 0 4px 20px rgba(0, 245, 255, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 10px 35px rgba(121, 40, 202, 0.7), 0 0 25px rgba(0, 245, 255, 0.7) !important;
        color: #ffffff !important;
    }

    /* Metric Cards Luxury Edition */
    [data-testid="stMetric"] {
        background: rgba(13, 17, 28, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
        padding: 18px 22px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 245, 255, 0.6) !important;
        box-shadow: 0 12px 35px rgba(0, 245, 255, 0.2) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00F5FF, #FF007A) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    /* Sidebar Navigation with Neon Highlight */
    div[data-testid="stSidebar"] {
        background-color: #07090e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 14px !important;
        padding: 11px 18px !important;
        margin-bottom: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(0, 245, 255, 0.1) !important;
        border-color: rgba(0, 245, 255, 0.5) !important;
        transform: translateX(6px) !important;
        box-shadow: 0 4px 20px rgba(0, 245, 255, 0.2) !important;
    }

    /* Glowing Pulsing Status */
    .pulse-online {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    .pulse-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 15px #10B981;
        animation: pulseAnimation 1.8s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.35); opacity: 1; box-shadow: 0 0 20px #10B981; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 22px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7928CA 0%, #00F5FF 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0, 245, 255, 0.4) !important;
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 35px 0 15px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 60px;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2. Dataset Definitions & Helpers
# ----------------------------------------------------------------------------
LOADERS = {
    "Diabetes": load_diabetes,
    "California Housing": fetch_california_housing,
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
}
TASK_DATASETS = {
    "regresi": ["Diabetes", "California Housing"],
    "klasifikasi": ["Iris", "Wine", "Breast Cancer"],
    "clustering": ["Iris", "Wine", "Breast Cancer"],
}
TASK_LABEL = {
    "regresi": "Regresi Linear",
    "klasifikasi": "Klasifikasi",
    "clustering": "Clustering",
}
DATASET_DESC = {
    "Diabetes": "Prediksi tingkat progres penyakit diabetes dalam 1 tahun berdasarkan metrik kesehatan.",
    "California Housing": "Prediksi rata-rata harga rumah di California berdasarkan fitur geografis & demografis.",
    "Iris": "Klasifikasi spesies bunga Iris berdasarkan ukuran panjang dan lebar kelopak.",
    "Wine": "Klasifikasi asal usul wine berdasarkan analisis kandungan kimia.",
    "Breast Cancer": "Klasifikasi tumor jinak/ganas berdasarkan fitur sel medis.",
}

def now_wib(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    return (dt + timedelta(hours=7)).strftime("%d %b %Y %H:%M WIB")

# ----------------------------------------------------------------------------
# 3. MongoDB & Groq Connection
# ----------------------------------------------------------------------------
def _get_secret(key_name, default=None):
    try:
        if key_name in st.secrets and st.secrets[key_name]:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)

@st.cache_resource(show_spinner=False)
def init_db():
    if not HAS_PYMONGO:
        return None, "Library pymongo belum terpasang."
    uri = _get_secret("MONGODB_URI")
    if not uri and "custom_mongo_uri" in st.session_state:
        uri = st.session_state.custom_mongo_uri
    if not uri:
        return None, "Koneksi MongoDB belum diatur."
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        return client[_get_secret("MONGODB_DB", "tugas_ai")], None
    except Exception as e:
        return None, str(e)

db, db_error = init_db()

# In-memory Mock Data jika MongoDB belum terhubung
if "mock_dataset_tambahan" not in st.session_state:
    st.session_state.mock_dataset_tambahan = []
if "mock_komentar" not in st.session_state:
    st.session_state.mock_komentar = [
        {
            "nama": "Dr. Suryasatriya Trihandaru, M.Sc.nat",
            "komentar": "Aplikasi interaktif pemodelan machine learning dan dashboard tersusun sangat rapi dan menarik!",
            "sentiment": "Positif 😊",
            "ai_reply": "Terima kasih banyak atas apresiasi dan bimbingannya, Pak Surya!",
            "created_at": datetime.now(timezone.utc) - timedelta(hours=2)
        }
    ]
if "mock_models" not in st.session_state:
    st.session_state.mock_models = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Halo! Saya adalah Asisten AI berbasis **Groq LPU (Llama 3.3)**. Tanyakan apa saja seputar model Regresi, Klasifikasi, Clustering, atau arsitektur sistem ini!"}
    ]

# ----------------------------------------------------------------------------
# 4. Helper Database & Groq Functions
# ----------------------------------------------------------------------------
def get_groq_api_key():
    key = _get_secret("GROQ_API_KEY")
    if not key and "custom_groq_key" in st.session_state:
        key = st.session_state.custom_groq_key
    return key

def analyze_comment_ai(comment_text):
    api_key = get_groq_api_key()
    if not api_key or not HAS_GROQ:
        return {"sentiment": "Positif 😊", "reply": "Terima kasih atas ulasan dan masukan Anda!"}
    try:
        client = Groq(api_key=api_key)
        prompt = (
            f'Analisis komentar berikut: "{comment_text}"\n'
            'Berikan respon JSON murni tanpa markdown (tanpa ```json) dengan format:\n'
            '{"sentiment": "Positif 😊" atau "Netral 😐" atau "Negatif 😡", "reply": "Balasan sopan dalam 1-2 kalimat bahasa Indonesia"}'
        )
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"sentiment": "Positif 😊", "reply": "Terima kasih banyak atas komentar dan partisipasi Anda!"}

@st.cache_data
def load_base(name):
    d = LOADERS[name](as_frame=True)
    feats = list(d.feature_names)
    tn = getattr(d, "target_names", None)
    tn = [str(x) for x in tn] if tn is not None else None
    if name == "California Housing":
        feats = feats[:6]
    elif name == "Breast Cancer":
        feats = feats[:8]
    df = d.frame[feats + ["target"]].copy()
    return df, feats, tn

def get_extra(task, name, feats):
    cols = feats + ["target"]
    if db is not None:
        try:
            docs = list(db.dataset_tambahan.find({"key": f"{task}:{name}"}))
            rows = [list(d["values"]) + [d.get("target")] for d in docs if len(d.get("values", [])) == len(feats)]
            return pd.DataFrame(rows, columns=cols).astype(float)
        except Exception:
            pass
    # Fallback mock
    docs = [d for d in st.session_state.mock_dataset_tambahan if d.get("key") == f"{task}:{name}"]
    rows = [list(d["values"]) + [d.get("target")] for d in docs if len(d.get("values", [])) == len(feats)]
    return pd.DataFrame(rows, columns=cols).astype(float) if rows else pd.DataFrame(columns=cols)

def get_training_df(task, name, use_extra=True):
    base, feats, tn = load_base(name)
    extra = get_extra(task, name, feats) if use_extra else pd.DataFrame(columns=feats + ["target"])
    if task != "clustering":
        extra = extra.dropna(subset=["target"])
    df = pd.concat([base, extra], ignore_index=True) if len(extra) else base.copy()
    if task == "klasifikasi":
        df["target"] = df["target"].astype(int)
    return df, len(extra), feats, tn

def save_model_record(task, dataset_name, model_obj, model_name, metrics):
    doc = {
        "key": f"{task}:{dataset_name}",
        "task": task,
        "dataset": dataset_name,
        "model_name": model_name,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc),
    }
    if db is not None:
        try:
            db.models.insert_one(doc)
            return True
        except Exception:
            pass
    st.session_state.mock_models.append(doc)
    return True

# ----------------------------------------------------------------------------
# 5. Navigasi Router
# ----------------------------------------------------------------------------
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "🏠 Dashboard Beranda"

def change_page(page_name):
    st.session_state["nav_radio"] = page_name

def feature_inputs(base, feats, prefix, ncols=3):
    vals = []
    cols = st.columns(ncols)
    for i, f in enumerate(feats):
        lo, hi = float(base[f].min()), float(base[f].max())
        pad = (hi - lo) * 0.5
        with cols[i % ncols]:
            v = st.number_input(
                f, value=float(base[f].median()),
                min_value=lo - pad, max_value=hi + pad,
                step=(hi - lo) / 100 or 0.01, format="%.4f",
                key=f"{prefix}_{f}",
            )
        vals.append(v)
    return vals

# ============================================================================
# HALAMAN 1: DASHBOARD BERANDA (DESAIN HEDON MODERN)
# ============================================================================
def page_home():
    st.markdown("""
        <div class="hero-title-hedon">🧠🤖 Dashboard Machine Learning</div>
        <div class="hero-subtitle-hedon">
            Aplikasi interaktif pemodelan <b>Regresi Linear</b>, <b>Klasifikasi</b>, dan <b>Clustering</b> terintegrasi <i>real-time</i> dengan <b>MongoDB Atlas</b> dan akselerasi AI <b>Groq LPU</b>.
        </div>
    """, unsafe_allow_html=True)

    # Grid 3 Kartu Modul Utama Bergaya Hologram Hedon
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('''
            <div class="hedon-card">
                <span class="card-badge badge-reg">SUPERVISED MODEL</span>
                <div class="hedon-title">📈 Regresi Linear</div>
                <div class="hedon-desc">
                    Prediksi nilai kontinu dengan dataset <b>Diabetes</b> & <b>California Housing</b>.
                </div>
            </div>
        ''', unsafe_allow_html=True)
        st.button("🚀 Modul Regresi", key="btn_nav_reg", use_container_width=True, on_click=change_page, args=("📈 Regresi Linear",))

    with c2:
        st.markdown('''
            <div class="hedon-card">
                <span class="card-badge badge-clf">MULTI-ALGORITHM</span>
                <div class="hedon-title">🧪 Klasifikasi Multi-Algoritma</div>
                <div class="hedon-desc">
                    Prediksi kategori dengan dataset <b>Iris</b>, <b>Wine</b>, & <b>Breast Cancer</b>.
                </div>
            </div>
        ''', unsafe_allow_html=True)
        st.button("🚀 Modul Klasifikasi", key="btn_nav_clf", use_container_width=True, on_click=change_page, args=("🧪 Klasifikasi",))

    with c3:
        st.markdown('''
            <div class="hedon-card">
                <span class="card-badge badge-clu">UNSUPERVISED AI</span>
                <div class="hedon-title">🔵 Clustering K-Means</div>
                <div class="hedon-desc">
                    Pengelompokan data tak berlabel dengan dataset <b>Iris</b>, <b>Wine</b>, & <b>Breast Cancer</b>.
                </div>
            </div>
        ''', unsafe_allow_html=True)
        st.button("🚀 Modul Clustering", key="btn_nav_clu", use_container_width=True, on_click=change_page, args=("🔵 Clustering",))

    st.write("")
    st.write("---")

    col_stat_title, col_stat_badge = st.columns([2, 1])
    with col_stat_title:
        st.markdown("### ⚡ Status Server & Database")
    with col_stat_badge:
        if db is not None:
            st.markdown("""
                <div style="text-align: right;">
                    <div class="pulse-online">
                        <span class="pulse-dot"></span> MongoDB Atlas terhubung aktif.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: right;">
                    <span style="background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                        🔴 MongoDB Belum Terhubung (Mode Lokal)
                    </span>
                </div>
            """, unsafe_allow_html=True)

    # Tabel Ringkasan Modul & Dataset
    rows = []
    for task, names in TASK_DATASETS.items():
        for n in names:
            n_extra = db.dataset_tambahan.count_documents({"key": f"{task}:{n}"}) if db is not None else len([d for d in st.session_state.mock_dataset_tambahan if d.get("key") == f"{task}:{n}"])
            n_models = db.models.count_documents({"key": f"{task}:{n}"}) if db is not None else len([d for d in st.session_state.mock_models if d.get("key") == f"{task}:{n}"])
            rows.append({
                "Modul": TASK_LABEL[task],
                "Dataset": n,
                "Deskripsi": DATASET_DESC.get(n, "-"),
                "Data Tambahan": n_extra,
                "Model Tersimpan": n_models,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    n_comments = db.komentar.count_documents({}) if db is not None else len(st.session_state.mock_komentar)
    n_total_models = db.models.count_documents({}) if db is not None else len(st.session_state.mock_models)
    st.caption(f"💬 Total Komentar Publik: **{n_comments}** | 📦 Total Model Tersimpan: **{n_total_models}**")

    st.markdown('<div class="footer">Dashboard Machine Learning Studio &bull; Cloud Database Atlas</div>', unsafe_allow_html=True)

# ============================================================================
# HALAMAN 2: REGRESI LINEAR
# ============================================================================
def page_regresi():
    st.markdown('<h1 class="main-title-container">📈 <span class="gradient-text">Regresi Linear & Lanjutan</span></h1>', unsafe_allow_html=True)
    st.info("💡 **Regresi** memprediksi nilai target berupa **angka kontinu** berdasarkan hubungan antara fitur input dan target.")

    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Pilih Dataset Regresi", TASK_DATASETS["regresi"], key="reg_ds")
    use_extra = c2.checkbox("Sertakan data tambahan dari MongoDB saat training", value=True, key="reg_extra")
    
    st.caption(f"ℹ️ **Deskripsi Dataset:** {DATASET_DESC.get(name, '')}")
    df, n_extra, feats, _ = get_training_df("regresi", name, use_extra)
    st.caption(f"📊 Total Data Latih: **{len(df)}** baris (Data Tambahan: **{n_extra}**)")

    c_m1, c_m2 = st.columns(2)
    model_name = c_m1.selectbox("Pilih Model Regresi", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest Regressor"])
    test_size = c_m2.slider("Proporsi Data Uji (Test Size)", 0.1, 0.5, 0.2, 0.05, key="reg_ts")

    # Training
    X, y = df.drop(columns="target"), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_name == "Linear Regression":
        m = LinearRegression()
    elif model_name == "Ridge Regression":
        m = Ridge(alpha=1.0)
    elif model_name == "Lasso Regression":
        m = Lasso(alpha=1.0)
    else:
        m = RandomForestRegressor(n_estimators=100, random_state=42)

    with st.spinner("Melatih model regresi..."):
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)

    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5

    # Metrik Cards
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score", f"{r2:.4f}")
    mc2.metric("MAE", f"{mae:.4f}")
    mc3.metric("RMSE", f"{rmse:.4f}")

    t1, t2, t3 = st.tabs(["📊 Dataset Overview", "📈 Grafik Actual vs Predicted", "🔮 Simulator Prediksi"])
    with t1:
        st.dataframe(df.head(25), use_container_width=True)
        st.write("**Statistik Deskriptif:**")
        st.dataframe(df.describe().T, use_container_width=True)

    with t2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yte.values, y=pred, mode="markers",
            marker=dict(size=8, color="#38bdf8", opacity=0.7),
            name="Data Prediksi",
            hovertemplate="Aktual: %{x:.2f}<br>Prediksi: %{y:.2f}<extra></extra>",
        ))
        min_v, max_v = min(yte.min(), pred.min()), max(yte.max(), pred.max())
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode="lines",
            line=dict(color="#f43f5e", dash="dash", width=2),
            name="Garis Ideal (y = x)",
        ))
        fig.update_layout(title="Aktual vs Prediksi", xaxis_title="Nilai Aktual", yaxis_title="Nilai Prediksi",
                          template="plotly_dark", height=430)
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.markdown(f"##### Masukkan Fitur untuk Simulasi Prediksi ({name}):")
        base = load_base(name)[0]
        vals = feature_inputs(base, feats, "reg_sim")
        if st.button("🚀 Prediksi Nilai Kontinu", type="primary", key="btn_sim_reg"):
            pred_single = float(m.predict(pd.DataFrame([vals], columns=feats))[0])
            st.success(f"🎯 **Hasil Estimasi Prediksi:** `{pred_single:.4f}`")

# ============================================================================
# HALAMAN 3: KLASIFIKASI
# ============================================================================
def page_klasifikasi():
    st.markdown('<h1 class="main-title-container">🧪 <span class="gradient-text">Klasifikasi Multi-Algoritma</span></h1>', unsafe_allow_html=True)
    st.info("💡 **Klasifikasi** memprediksi label kategori target (misal: spesies bunga, diagnosa) berdasarkan fitur data.")

    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Pilih Dataset Klasifikasi", TASK_DATASETS["klasifikasi"], key="clf_ds")
    use_extra = c2.checkbox("Sertakan data tambahan dari MongoDB saat training", value=True, key="clf_extra")
    
    st.caption(f"ℹ️ **Deskripsi Dataset:** {DATASET_DESC.get(name, '')}")
    df, n_extra, feats, tn = get_training_df("klasifikasi", name, use_extra)
    st.caption(f"📊 Total Data Latih: **{len(df)}** baris (Data Tambahan: **{n_extra}**)")

    c_m1, c_m2 = st.columns(2)
    model_name = c_m1.selectbox("Pilih Algoritma", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"])
    test_size = c_m2.slider("Proporsi Data Uji", 0.1, 0.5, 0.2, 0.05, key="clf_ts")

    X, y = df.drop(columns="target"), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if model_name == "Logistic Regression":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif model_name == "KNN":
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

    with st.spinner("Melatih model klasifikasi..."):
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro", zero_division=0)
    rec = recall_score(yte, pred, average="macro", zero_division=0)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)

    # Metrik Cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Akurasi", f"{acc * 100:.2f}%")
    mc2.metric("Precision", f"{prec:.4f}")
    mc3.metric("Recall", f"{rec:.4f}")
    mc4.metric("F1 Score", f"{f1:.4f}")

    t1, t2, t3 = st.tabs(["📊 Distribusi Data", "🎯 Confusion Matrix", "🔮 Simulator Prediksi"])
    with t1:
        show = df.copy()
        show["Kelas"] = show["target"].map(lambda i: tn[int(i)] if tn else str(int(i)))
        st.dataframe(show.head(25), use_container_width=True)
        vc = show["Kelas"].value_counts().reset_index()
        vc.columns = ["Kelas", "Jumlah"]
        fig = px.bar(vc, x="Kelas", y="Jumlah", color="Kelas", title=f"Distribusi Kelas ({name})", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        classes = sorted(y.unique())
        labels_display = [tn[c] if tn else str(c) for c in classes]
        cm = confusion_matrix(yte, pred, labels=classes)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=labels_display, y=labels_display,
            color_continuous_scale="Blues", template="plotly_dark", height=420
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with t3:
        st.markdown(f"##### Masukkan Fitur untuk Simulasi Klasifikasi ({name}):")
        base = load_base(name)[0]
        vals = feature_inputs(base, feats, "clf_sim")
        if st.button("🚀 Prediksi Kelas Target", type="primary", key="btn_sim_clf"):
            pred_c = int(clf.predict(pd.DataFrame([vals], columns=feats))[0])
            lbl_res = tn[pred_c] if tn else str(pred_c)
            st.balloons()
            st.success(f"🎯 **Hasil Prediksi Kelas:** `{lbl_res}`")

# ============================================================================
# HALAMAN 4: CLUSTERING (K-MEANS)
# ============================================================================
def page_clustering():
    st.markdown('<h1 class="main-title-container">🔵 <span class="gradient-text">Clustering K-Means</span></h1>', unsafe_allow_html=True)
    st.info("💡 **Clustering** mengelompokkan data tanpa label awal. Objek yang berada dalam satu klaster memiliki kemiripan pola karakteristik yang tinggi.")

    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Pilih Dataset Clustering", TASK_DATASETS["clustering"], key="clu_ds")
    use_extra = c2.checkbox("Sertakan data tambahan dari MongoDB", value=True, key="clu_extra")
    
    st.caption(f"ℹ️ **Deskripsi Dataset:** {DATASET_DESC.get(name, '')}")
    df, n_extra, feats, _ = get_training_df("clustering", name, use_extra)
    k = st.slider("Jumlah Cluster (k)", 2, 8, 3, key="clu_k")

    X = df.drop(columns="target", errors="ignore")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    with st.spinner("Menjalankan K-Means Clustering..."):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        labels = km.labels_
        sil = silhouette_score(Xs, labels)
        db_idx = davies_bouldin_score(Xs, labels)
        pca_res = PCA(n_components=2, random_state=42).fit_transform(Xs)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Silhouette Score", f"{sil:.4f}", help="Mendekati 1.0 berarti pemisahan klaster sangat baik")
    mc2.metric("Davies-Bouldin Index", f"{db_idx:.4f}", help="Semakin kecil nilainya semakin baik")
    mc3.metric("Inertia", f"{km.inertia_:,.1f}")

    t1, t2, t3 = st.tabs(["🎨 Proyeksi Klaster (PCA 2D)", "📊 Karakteristik Klaster", "🔮 Prediksi Klaster Baru"])
    with t1:
        pca_df = pd.DataFrame(pca_res, columns=["PC1", "PC2"])
        pca_df["Cluster"] = [f"Cluster {l}" for l in labels]
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            title=f"Proyeksi PCA 2D — K-Means (k={k}, {name})",
            template="plotly_dark", height=440
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with t2:
        df_clust = X.copy()
        df_clust["Cluster"] = labels
        c_a, c_b = st.columns(2)
        vc = df_clust["Cluster"].value_counts().sort_index().reset_index()
        vc.columns = ["Cluster", "Jumlah Data"]
        fig_b = px.bar(vc, x="Cluster", y="Jumlah Data", color="Cluster", title="Jumlah Sampel per Cluster", template="plotly_dark")
        c_a.plotly_chart(fig_b, use_container_width=True)
        c_b.write("**Nilai Rata-rata Fitur per Cluster:**")
        c_b.dataframe(df_clust.groupby("Cluster").mean().style.format("{:.3f}"), use_container_width=True)

    with t3:
        st.markdown(f"##### Masukkan Fitur untuk Menentukan Cluster ({name}):")
        base = load_base(name)[0]
        vals = feature_inputs(base, feats, "clu_sim")
        if st.button("🚀 Tentukan Cluster", type="primary", key="btn_sim_clu"):
            X_new = sc.transform(pd.DataFrame([vals], columns=feats))
            cid = int(km.predict(X_new)[0])
            st.success(f"🎯 Data ini masuk ke dalam: **Cluster {cid}**")

# ============================================================================
# HALAMAN 5: UPDATE DATA & RETRAINING (MONGODB ATLAS)
# ============================================================================
def page_retrain():
    st.markdown('<h1 class="main-title-container">🔁 <span class="gradient-text">Management Data & Retraining</span></h1>', unsafe_allow_html=True)
    st.info("Tambahkan data baru ke MongoDB Atlas, kelola sampel, dan jalankan **Retraining** untuk memperbarui model secara *real-time*.")

    c1, c2 = st.columns(2)
    task = c1.selectbox("Modul Pemodelan", list(TASK_LABEL), format_func=TASK_LABEL.get, key="rt_task")
    name = c2.selectbox("Dataset Tujuan", TASK_DATASETS[task], key="rt_ds")

    base, feats, tn = load_base(name)
    key = f"{task}:{name}"
    
    n_extra = db.dataset_tambahan.count_documents({"key": key}) if db is not None else len([d for d in st.session_state.mock_dataset_tambahan if d.get("key") == key])
    st.caption(f"📁 Data Asli Sklearn: **{len(base)}** baris | 📥 Data Tambahan Tersimpan: **{n_extra}** baris")

    t1, t2, t3 = st.tabs(["➕ Input Data Baru", "🗂️ Kelola Data", "🔁 Retraining Model"])

    with t1:
        st.markdown(f"##### Masukkan Sampel Baru ({name}):")
        with st.form("form_add_data", clear_on_submit=True):
            vals = feature_inputs(base, feats, "rt_add")
            target_val = None
            if task == "regresi":
                target_val = st.number_input("Target Value (Kontinu):", value=float(base["target"].median()))
            elif task == "klasifikasi":
                opts = list(range(len(tn))) if tn else [0, 1]
                target_val = st.selectbox("Target Kelas:", opts, format_func=lambda x: tn[x] if tn else str(x))

            if st.form_submit_button("💾 Simpan Data ke MongoDB"):
                new_doc = {
                    "key": key,
                    "task": task,
                    "dataset": name,
                    "values": [float(v) for v in vals],
                    "target": float(target_val) if target_val is not None else None,
                    "created_at": datetime.now(timezone.utc),
                }
                if db is not None:
                    try:
                        db.dataset_tambahan.insert_one(new_doc)
                        st.success("✅ Data berhasil disimpan ke MongoDB Atlas!")
                    except Exception as e:
                        st.error(f"Gagal simpan ke MongoDB: {e}")
                else:
                    st.session_state.mock_dataset_tambahan.append(new_doc)
                    st.info("ℹ️ Disimpan ke Local Storage (MongoDB belum terhubung)")
                time.sleep(0.5)
                st.rerun()

    with t2:
        docs = []
        if db is not None:
            try:
                docs = list(db.dataset_tambahan.find({"key": key}).sort("created_at", -1))
            except Exception:
                pass
        else:
            docs = [d for d in st.session_state.mock_dataset_tambahan if d.get("key") == key]

        if docs:
            rows_display = []
            for d in docs:
                row_dict = {"ID": str(d.get("_id", len(rows_display)+1))}
                for idx, f in enumerate(feats):
                    if idx < len(d.get("values", [])):
                        row_dict[f] = d["values"][idx]
                row_dict["Target"] = d.get("target")
                rows_display.append(row_dict)
            st.dataframe(pd.DataFrame(rows_display), use_container_width=True)

            del_id = st.selectbox("Pilih ID Data untuk Dihapus:", [r["ID"] for r in rows_display])
            if st.button("🗑️ Hapus Data Ini"):
                if db is not None:
                    try:
                        db.dataset_tambahan.delete_one({"_id": ObjectId(del_id)})
                        st.success("Data berhasil dihapus dari MongoDB Atlas!")
                    except Exception as e:
                        st.error(f"Gagal hapus: {e}")
                else:
                    st.session_state.mock_dataset_tambahan = [d for d in st.session_state.mock_dataset_tambahan if str(d.get("_id", "")) != del_id]
                    st.success("Data dihapus dari memori!")
                time.sleep(0.5)
                st.rerun()
        else:
            st.info("Belum ada data tambahan untuk dataset ini.")

    with t3:
        st.markdown("##### Jalankan Retraining Menggunakan Semua Data")
        st.write("Menggabungkan data asli + data baru, melatih ulang model, dan memperbarui metrik secara instan.")
        if st.button("🚀 Retrain Model Sekarang", type="primary"):
            df_full, n_ex, _, _ = get_training_df(task, name, use_extra=True)
            with st.spinner("Melatih ulang model..."):
                if task == "klasifikasi":
                    X_f, y_f = df_full.drop(columns="target"), df_full["target"]
                    Xtr, Xte, ytr, yte = train_test_split(X_f, y_f, test_size=0.25, random_state=42)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
                    new_acc = accuracy_score(yte, rf.predict(Xte))
                    metrics = {"Akurasi": new_acc}
                elif task == "regresi":
                    X_f, y_f = df_full.drop(columns="target"), df_full["target"]
                    Xtr, Xte, ytr, yte = train_test_split(X_f, y_f, test_size=0.25, random_state=42)
                    rg = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
                    new_r2 = r2_score(yte, rg.predict(Xte))
                    metrics = {"R² Score": new_r2}
                else:
                    Xs_f = StandardScaler().fit_transform(df_full.drop(columns="target", errors="ignore"))
                    km = KMeans(n_clusters=3, random_state=42).fit(Xs_f)
                    metrics = {"Silhouette": silhouette_score(Xs_f, km.labels_)}

                save_model_record(task, name, None, "Random Forest", metrics)
                st.balloons()
                st.success(f"🎉 **Retraining Berhasil!** Model diperbarui dengan total **{len(df_full)} sampel**.")
                for k_met, v_met in metrics.items():
                    st.metric(f"Metrik Terbaru ({k_met})", f"{v_met:.4f}")

# ============================================================================
# HALAMAN 6: DISKUSI KOMENTAR (INTEGRASI GROQ AI)
# ============================================================================
def page_komentar():
    st.markdown('<h1 class="main-title-container">💬 <span class="gradient-text">Diskusi & Komentar Publik</span></h1>', unsafe_allow_html=True)
    st.markdown("Fitur publik untuk berbagi ulasan. Setiap komentar otomatis dianalisis oleh **Groq AI (Llama 3.3)** untuk mendeteksi sentimen dan memberikan balasan ramah.")

    c1, c2 = st.columns([1.1, 1.6])
    with c1:
        st.markdown("#### ✍️ Tulis Komentar")
        with st.form("form_komentar", clear_on_submit=True):
            nama = st.text_input("Nama Pengunjung:", placeholder="Misal: Budi Santoso")
            komentar_text = st.text_area("Komentar / Ulasan:", placeholder="Tuliskan ulasan Anda mengenai dashboard ini...")
            if st.form_submit_button("🚀 Kirim Komentar"):
                if nama.strip() and komentar_text.strip():
                    with st.spinner("🤖 Groq AI sedang menganalisis sentimen..."):
                        res_ai = analyze_comment_ai(komentar_text)
                    doc = {
                        "nama": nama.strip(),
                        "komentar": komentar_text.strip(),
                        "sentiment": res_ai.get("sentiment", "Positif 😊"),
                        "ai_reply": res_ai.get("reply", "Terima kasih atas ulasan Anda!"),
                        "created_at": datetime.now(timezone.utc),
                    }
                    if db is not None:
                        try:
                            db.komentar.insert_one(doc)
                            st.success("✅ Komentar tersimpan di MongoDB Atlas!")
                        except Exception as e:
                            st.error(f"Gagal simpan: {e}")
                    else:
                        st.session_state.mock_komentar.insert(0, doc)
                        st.success("✅ Komentar tersimpan di memori!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("Mohon isi Nama dan Komentar terlebih dahulu.")

    with c2:
        st.markdown("#### 💬 Tanggapan & Ulasan Pengunjung")
        docs = []
        if db is not None:
            try:
                docs = list(db.komentar.find().sort("created_at", -1).limit(40))
            except Exception:
                pass
        else:
            docs = st.session_state.mock_komentar

        if docs:
            for d in docs:
                sentimen = d.get("sentiment", "Positif 😊")
                sent_color = "#10B981" if "Positif" in sentimen else ("#F59E0B" if "Netral" in sentimen else "#EF4444")
                with st.chat_message("user"):
                    st.markdown(f"**{d.get('nama', 'Pengunjung')}** &bull; <span style='background:{sent_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem;'>Sentimen: {sentimen}</span>", unsafe_allow_html=True)
                    st.write(d.get("komentar", ""))
                    if d.get("ai_reply"):
                        st.markdown(f"""
                        <div class="ai-reply-box">
                            <strong>🤖 Respon Asisten AI (Groq):</strong><br>
                            <span>{d.get('ai_reply')}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Belum ada komentar tersimpan. Jadilah yang pertama memberikan ulasan!")

# ============================================================================
# HALAMAN 7: ASISTEN GROQ AI
# ============================================================================
def page_groq():
    st.markdown('<h1 class="main-title-container">🤖 <span class="gradient-text">Asisten Cerdas Groq AI</span></h1>', unsafe_allow_html=True)
    st.caption("Konsultasi interaktif seputar Machine Learning, Algoritma, Evaluasi, dan Arsitektur Cloud menggunakan inferensi super cepat Llama 3.3.")

    groq_key = get_groq_api_key()

    # Quick Prompts
    st.markdown("##### 💡 Contoh Pertanyaan Cepat:")
    q1, q2, q3 = st.columns(3)
    quick_query = None
    if q1.button("📈 Kapan pakai Regresi vs Klasifikasi?", use_container_width=True):
        quick_query = "Kapan kita sebaiknya menggunakan model Regresi dan kapan menggunakan Klasifikasi? Berikan contohnya."
    if q2.button("🔮 Apa fungsi Silhouette Score pada Clustering?", use_container_width=True):
        quick_query = "Jelaskan prinsip kerja K-Means Clustering dan metrik evaluasi Silhouette Score!"
    if q3.button("🍃 Mengapa butuh MongoDB di Web ML?", use_container_width=True):
        quick_query = "Mengapa kita menghubungkan MongoDB Atlas ke Streamlit untuk dataset dan komentar publik?"

    # Render History
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Tanyakan apa saja seputar Machine Learning...")
    actual_q = quick_query if quick_query else user_q

    if actual_q:
        st.session_state.chat_history.append({"role": "user", "content": actual_q})
        with st.chat_message("user"):
            st.markdown(actual_q)

        with st.chat_message("assistant"):
            if groq_key and HAS_GROQ:
                try:
                    client = Groq(api_key=groq_key)
                    sys_msg = "Anda adalah asisten AI ramah dan edukatif seputar Machine Learning, Regresi, Klasifikasi, Clustering, dan MongoDB Atlas."
                    payload = [{"role": "system", "content": sys_msg}] + [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-6:]
                    ]
                    stream = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=payload,
                        temperature=0.5,
                        max_tokens=600,
                        stream=True
                    )
                    ans = st.write_stream(stream)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as err:
                    fallback_msg = f"Kendala koneksi Groq: `{err}`. Pastikan API key valid."
                    st.error(fallback_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": fallback_msg})
            else:
                fallback_msg = (
                    "**Jawaban Asisten:**\n\n"
                    "- **Regresi** digunakan saat target bernilai kontinu (misal harga rumah, kadar glukosa).\n"
                    "- **Klasifikasi** digunakan saat target berupa kategori diskrit (misal spesies Iris, deteksi kanker ganas/jinak).\n"
                    "- **Clustering** adalah *unsupervised learning* untuk mengelompokkan data tanpa label berdasarkan kemiripan pola fitur.\n"
                    "- **MongoDB Atlas** berguna untuk menyimpan data sampel baru secara cloud, sehingga model dapat di-**retrain** secara real-time!\n\n"
                    "*(💡 Masukkan `GROQ_API_KEY` di `.streamlit/secrets.toml` untuk mengaktifkan streaming interaktif Llama 3.3)*"
                )
                st.markdown(fallback_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": fallback_msg})

# ============================================================================
# 6. SIDEBAR NAVIGASI & ROUTER (HEDON / PRO DEVELOPER)
# ============================================================================
PAGES = {
    "🏠 Dashboard Beranda": page_home,
    "📈 Regresi Linear": page_regresi,
    "🧪 Klasifikasi": page_klasifikasi,
    "🔵 Clustering": page_clustering,
    "🔁 Update Data & Retraining": page_retrain,
    "💬 Diskusi Komentar": page_komentar,
    "🤖 Asisten Groq AI": page_groq,
}

# Branding Header Sidebar
st.sidebar.markdown("""
    <div style="padding: 8px 0 18px 0;">
        <div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 2px; color: #00F5FF; text-transform: uppercase;">
            🧠🤖 MACHINE LEARNING
        </div>
        <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; font-family: 'Space Grotesk', sans-serif; margin-top: 2px;">
            DASHBOARD <span style="background: linear-gradient(90deg, #00F5FF, #FF007A); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ML</span>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500;">
            Aplikasi Interaktif &bull; Tugas Kuliah AI
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("###### 🧭 Menu Navigasi")
choice = st.sidebar.radio(
    "Navigasi",
    list(PAGES.keys()),
    label_visibility="collapsed",
    key="nav_radio"
)
st.sidebar.divider()

# Indikator Status Database di Bawah Sidebar
if db is None:
    st.sidebar.markdown("""
        <div style="background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); padding: 8px 12px; border-radius: 10px; font-size: 0.78rem; font-weight: bold; margin-bottom: 8px;">
            🔴 MongoDB: Mode Lokal
        </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("🔄 Hubungkan Ulang"):
        init_db.clear()
        st.rerun()
else:
    st.sidebar.markdown("""
        <div class="pulse-online" style="margin-bottom: 8px; width: 100%; justify-content: center;">
            <span class="pulse-dot"></span> MongoDB Atlas: Online
        </div>
    """, unsafe_allow_html=True)
    n_k = db.komentar.count_documents({})
    n_m = db.models.count_documents({})
    st.sidebar.caption(f"💬 **{n_k}** ulasan publik")
    st.sidebar.caption(f"📦 **{n_m}** model tersimpan")

groq_ready = bool(get_groq_api_key())
if groq_ready:
    st.sidebar.caption("⚡ **Groq LPU:** `Aktif`")
else:
    st.sidebar.caption("⚡ **Groq LPU:** `Standby`")

# Eksekusi Halaman Terpilih
PAGES[choice]()
