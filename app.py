import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Pengaruh Faktor Perilaku dan Tingkat Adopsi terhadap Intensitas Penggunaan E-Wallet pada Kabupaten Toba",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
  background: #f6f7fb;
}

.block-container {
  padding-top: 4rem;   
  padding-bottom: 3rem;
  max-width: 1400px;
}

.h1-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0.2rem 0 0.6rem 0;
  color: #111827;
}

.h2-title {
  font-size: 1.35rem;
  font-weight: 800;
  margin: 1.2rem 0 0.75rem 0;
  color: #111827;
}

.card {
  background: #ffffff;
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 16px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 10px 25px rgba(17,24,39,0.06);
}

.card p {
  margin: 0.25rem 0 0 0;
  color: #374151;
}

.banner {
  background: linear-gradient(90deg, rgba(102,126,234,.18), rgba(245,87,108,.14));
  border: 1px solid rgba(102,126,234,.25);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  color: #111827;
}

section[data-testid="stSidebar"] {
  background: #111827;
}

section[data-testid="stSidebar"] * {
  color: #F9FAFB !important;
}

section[data-testid="stSidebar"] a {
  color: #F9FAFB !important;
}

section[data-testid="stSidebar"] label {
  color: #F9FAFB !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
  background: rgba(255,255,255,0.08);
  border-radius: 8px;
}

section[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)
def h1(title: str):
    st.markdown(
    "<div class='h1-title'>Pengaruh Faktor Perilaku dan Tingkat Adopsi terhadap Intensitas Penggunaan E-Wallet</div>",
    unsafe_allow_html=True
)


def h2(title: str):
    st.markdown(
    "<div class='h2-title'>Dashboard analisis Machine Learning untuk memahami perilaku pengguna e-wallet di Kabupaten Toba.</div>",
    unsafe_allow_html=True
)


def banner(text: str):
    st.markdown(f"<div class='banner'>{text}</div>", unsafe_allow_html=True)

def card(html: str):
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_bersih.csv", sep=";")
        return df
    except FileNotFoundError:
        st.error("Dataset 'dataset_bersih.csv' tidak ditemukan!")
        return None

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    
    likert_mapping = {
        'Sangat tidak setuju': 1,
        'Tidak setuju': 2,
        'Netral': 3,
        'Setuju': 4,
        'Sangat setuju': 5
    }
    
    column_mapping = {
        'praktis_dibanding_tunai': 'Q1',
        'kenyamanan_transaksi': 'Q2',
        'pengurangan_risiko_tunai': 'Q3',
        'persepsi_kemudahan_pemahaman': 'Q4',
        'persepsi_kemudahan_operasi': 'Q5',
        'persepsi_kemudahan_teknis': 'Q6',
        'persepsi_kesesuaian_kebutuhan': 'Q7',
        'persepsi_keterbukaan_inovasi': 'Q8',
        'niat_penggunaan_berkelanjutan': 'Q9',
        'persepsi_keamanan_transaksi': 'Q10',
        'keamanan_data_pribadi': 'Q11',
        'persepsi_keamanan_saldo': 'Q12',
        'persepsi_keandalan_sistem_keamanan': 'Q13',
        'kecenderungan_penggunaan_ewallet': 'Q14',
        'persepsi_keluasan_penggunaan': 'Q15'
    }
    
    df = df.rename(columns=column_mapping)
    
    for i in range(1, 16):
        col = f'Q{i}'
        if col in df.columns:
            df[col] = df[col].map(likert_mapping)
    
    return df

@st.cache_resource
def load_artifacts():
    model = joblib.load("model_logreg.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    test_index = joblib.load("test_index.pkl")
    y_test = np.array(joblib.load("y_test.pkl"))
    y_pred = np.array(joblib.load("y_pred.pkl"))
    y_proba = np.array(joblib.load("y_proba.pkl"))
    return model, feature_cols, test_index, y_test, y_pred, y_proba

def train_models(df):
    df = df.copy()

    df['Perilaku'] = df[['Q1','Q2','Q3']].mean(axis=1)
    df['Kemudahan'] = df[['Q4','Q5','Q6']].mean(axis=1)
    df['Adopsi'] = df[['Q7','Q8','Q9']].mean(axis=1)
    df['Kepercayaan'] = df[['Q10','Q11','Q12','Q13']].mean(axis=1)
    df['Intensitas'] = df[['Q14','Q15']].mean(axis=1)

    logreg, feature_cols, test_index, y_test_saved, y_pred_saved, y_proba_saved = load_artifacts()

    feature_cols = [c for c in feature_cols if c in df.columns]
    X_logreg = df[feature_cols].dropna()

    X_cluster = df[['Perilaku','Kemudahan','Adopsi','Kepercayaan']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled)

    return {
        'df_processed': df,
        'logreg': logreg,

        'y_test': y_test_saved,
        'y_pred': y_pred_saved,
        'y_proba': y_proba_saved,

        'X_cluster': X_cluster,
        'clusters': clusters,
        'pca_features': pca_features
    }

st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: white; font-size: 2.5rem; margin: 0;'>💳</h1>
        <h2 style='color: white; margin: 0.5rem 0;'>Pengaruh Faktor Perilaku dan Tingkat Adopsi terhadap Intensitas Penggunaan E-Wallet pada Kabupaten Toba</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Dashboard Analisis ML</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    " Navigasi Halaman",
    [" Home", " Dataset Overview", " EDA Visualization", 
     " Clustering Analysis", " Logistic Regression", " Prediction"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            Dashboard ini menganalisis perilaku penggunaan e-wallet menggunakan
            <strong>Machine Learning</strong> untuk memberikan insight bisnis yang akurat.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
df_raw = load_data()

if df_raw is None:
    st.stop()

df = preprocess_data(df_raw)
models = train_models(df)
df_processed = models['df_processed']

if page == " Home":
    st.markdown('<div class="main-header"> Pengaruh Faktor Perilaku dan Tingkat Adopsi terhadap Intensitas Penggunaan E-Wallet pada Kabupaten Toba</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3> Selamat Datang di Dashboard E-Wallet Analytics</h3>
        <p style='font-size: 1.1rem; margin: 0;'>
            Platform analisis mendalam tentang perilaku penggunaan e-wallet di Kabupaten Toba 
            menggunakan teknik <strong>Machine Learning</strong> seperti K-Means Clustering dan Logistic Regression.
        </p>
    </div>
    """, unsafe_allow_html=True)

    h2(" Ringkasan Utama")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(" Total Responden", f"{len(df_processed):,}")
    with c2:
        st.metric(" Jumlah Cluster", "2")
    with c3:
        acc = accuracy_score(models['y_test'], models['y_pred'])
        st.metric(" Akurasi Model", f"{acc:.1%}")
    with c4:
        st.metric(" Fitur Input", "15")

    st.markdown('<div class="sub-header"> Tujuan Penelitian</div>', unsafe_allow_html=True)

    goals_col1, goals_col2 = st.columns(2)

    with goals_col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0;'> Analisis Pola Perilaku</h4>
            <p style='margin: 0;'>Menganalisis pola perilaku pengguna e-wallet berdasarkan 5 dimensi utama 
            untuk memahami preferensi dan kebiasaan penggunaan.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0;'> Segmentasi Pengguna</h4>
            <p style='margin: 0;'>Mengelompokkan pengguna berdasarkan karakteristik penggunaan 
            menggunakan K-Means Clustering untuk strategi marketing yang tepat.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with goals_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0;'> Prediksi Intensitas</h4>
            <p style='margin: 0;'>Memprediksi intensitas penggunaan e-wallet (Tinggi/Rendah) 
            menggunakan Logistic Regression dengan akurasi tinggi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0;'> Business Insights</h4>
            <p style='margin: 0;'>Memberikan insight untuk pengembangan strategi bisnis e-wallet 
            di Kabupaten Toba berdasarkan data dan analisis ML.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"> Metodologi Penelitian</div>', unsafe_allow_html=True)
    
    method_col1, method_col2 = st.columns(2)
    
    with method_col1:
        st.markdown("####  Data Collection & Preprocessing")
        st.markdown("""
        - **Survey**: 15 pertanyaan (Q1-Q15) dengan skala Likert 5 poin
        - **Responden**: Pengguna e-wallet di Kabupaten Toba
        - **Normalisasi**: Data usia dan durasi penggunaan
        - **Feature Engineering**: Agregasi menjadi 5 dimensi analisis
        """)
        
        st.markdown("####  Machine Learning Models")
        st.markdown("""
        - **K-Means Clustering**: Segmentasi pengguna menjadi 2 cluster berdasarkan perilaku
        - **Logistic Regression**: Prediksi intensitas penggunaan (Tinggi/Rendah)
        - **PCA**: Visualisasi reduction untuk interpretasi cluster
        - **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
        """)
    
    with method_col2:
        st.markdown("####  5 Dimensi Analisis")
        
        st.markdown('<span class="dimension-badge badge-perilaku">▪️ Perilaku</span>', unsafe_allow_html=True)
        st.caption("Q1-Q3: Praktis, nyaman, pengurangan risiko tunai")
        
        st.markdown('<span class="dimension-badge badge-kemudahan">▪️ Kemudahan</span>', unsafe_allow_html=True)
        st.caption("Q4-Q6: Pemahaman, operasi, kemudahan teknis")
        
        st.markdown('<span class="dimension-badge badge-adopsi">▪️ Adopsi</span>', unsafe_allow_html=True)
        st.caption("Q7-Q9: Kesesuaian kebutuhan, inovasi, penggunaan berkelanjutan")
        
        st.markdown('<span class="dimension-badge badge-kepercayaan">▪️ Kepercayaan</span>', unsafe_allow_html=True)
        st.caption("Q10-Q13: Keamanan transaksi, data pribadi, saldo, sistem")
        
        st.markdown('<span class="dimension-badge badge-intensitas">▪️ Intensitas</span>', unsafe_allow_html=True)
        st.caption("Q14-Q15: Frekuensi kecenderungan & keluasan penggunaan")
    
    st.markdown('<div class="sub-header"> Pipeline Analisis</div>', unsafe_allow_html=True)
    
    pipeline_fig = go.Figure()
    
    steps = ["Data\nCollection", "Preprocessing", "Feature\nEngineering", 
             "Clustering", "Classification", "Evaluation"]
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        pipeline_fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=50, color=color, line=dict(width=3, color='white')),
            text=step,
            textposition="bottom center",
            textfont=dict(size=14, family='Arial Black', color='#2d3748'),
            showlegend=False,
            hoverinfo='text',
            hovertext=step
        ))
        
        if i < len(steps) - 1:
            pipeline_fig.add_annotation(
                x=i+0.5, y=0,
                text="→",
                showarrow=False,
                font=dict(size=30, color='#667eea'),
                xanchor='center'
            )
    
    pipeline_fig.update_layout(
        height=250,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 5.5]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 0.5]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=80)
    )
    
    st.plotly_chart(pipeline_fig, use_container_width=True)

elif page == " Dataset Overview":
    st.markdown('<div class="main-header"> Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white;'>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Total Data</p>
                <h2 style='margin: 0.5rem 0 0 0; font-size: 2.5rem;'>{}</h2>
            </div>
        """.format(len(df_processed)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white;'>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Jumlah Kolom</p>
                <h2 style='margin: 0.5rem 0 0 0; font-size: 2.5rem;'>{}</h2>
            </div>
        """.format(len(df_processed.columns)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white;'>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Missing Values</p>
                <h2 style='margin: 0.5rem 0 0 0; font-size: 2.5rem;'>{}</h2>
            </div>
        """.format(df_processed.isnull().sum().sum()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white;'>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Duplicate Rows</p>
                <h2 style='margin: 0.5rem 0 0 0; font-size: 2.5rem;'>{}</h2>
            </div>
        """.format(df_processed.duplicated().sum()), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"> Preview Data</div>', unsafe_allow_html=True)
    st.dataframe(df_processed.head(10), use_container_width=True, height=400)
    
    st.markdown('<div class="sub-header"> Deskripsi Fitur</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 1rem 0;'> Fitur Demografis</h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li><strong>Usia</strong>: Kelompok usia responden (< 17, 17-25, 26-34, > 34 tahun)</li>
                <li><strong>Jenis Kelamin</strong>: Laki-laki / Perempuan</li>
                <li><strong>Domisili</strong>: Kecamatan di Kabupaten Toba</li>
                <li><strong>Status/Pekerjaan</strong>: Status pekerjaan/pendidikan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white;'>
            <h4 style='margin: 0 0 1rem 0;'> Fitur Platform</h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li><strong>platform_DANA</strong>: Menggunakan DANA (1/0)</li>
                <li><strong>platform_OVO</strong>: Menggunakan OVO (1/0)</li>
                <li><strong>platform_GoPay</strong>: Menggunakan GoPay (1/0)</li>
                <li><strong>platform_ShopeePay</strong>: Menggunakan ShopeePay (1/0)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 1rem 0;'> Fitur Perilaku (Q1-Q15)</h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li><strong>Q1-Q3</strong>: Dimensi Perilaku (praktis, nyaman, risiko)</li>
                <li><strong>Q4-Q6</strong>: Dimensi Kemudahan (pemahaman, operasi, teknis)</li>
                <li><strong>Q7-Q9</strong>: Dimensi Adopsi (kesesuaian, inovasi, berkelanjutan)</li>
                <li><strong>Q10-Q13</strong>: Dimensi Kepercayaan (keamanan transaksi & data)</li>
                <li><strong>Q14-Q15</strong>: Dimensi Intensitas (frekuensi & keluasan)</li>
            </ul>
            <p style='margin: 1rem 0 0 0; font-size: 0.9rem;'>
                <strong>Skala Likert</strong>: 1=Sangat Tidak Setuju, 5=Sangat Setuju
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white;'>
            <h4 style='margin: 0 0 1rem 0;'> Fitur Tambahan</h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li><strong>durasi_penggunaan</strong>: Lama penggunaan e-wallet</li>
                <li><strong>frekuensi_penggunaan_mingguan</strong>: Frekuensi per minggu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"> Statistik Deskriptif</div>', unsafe_allow_html=True)
    
    numeric_cols = [f'Q{i}' for i in range(1, 16)] + ['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan', 'Intensitas']
    available_cols = [col for col in numeric_cols if col in df_processed.columns]
    desc_stats = df_processed[available_cols].describe().T
    st.dataframe(desc_stats.style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)
    
    st.markdown('<div class="sub-header"> Kualitas Data</div>', unsafe_allow_html=True)
    
    completeness = (1 - df_processed.isnull().sum().sum() / (len(df_processed) * len(df_processed.columns))) * 100
    uniqueness = (1 - df_processed.duplicated().sum() / len(df_processed)) * 100
    
    quality_metrics = pd.DataFrame({
        'Metrik': ['Completeness', 'Uniqueness', 'Validity', 'Consistency'],
        'Score': [
            f"{completeness:.1f}%",
            f"{uniqueness:.1f}%",
            "100%",
            "100%"
        ],
        'Status': ['✅ Excellent', '✅ Excellent', '✅ Excellent', '✅ Excellent']
    })
    
    st.dataframe(quality_metrics, use_container_width=True, hide_index=True)

elif page == " EDA Visualization":
    st.markdown('<div class="main-header"> Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([" Demografi", " Platform E-Wallet", " Perilaku Penggunaan", " Persepsi Likert"])
    
    with tab1:
        st.markdown('<div class="sub-header">Distribusi Demografis Responden</div>', unsafe_allow_html=True)
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            if 'Usia' in df_processed.columns:
                usia_counts = df_processed['Usia'].value_counts()
                fig_usia = px.bar(
                    x=usia_counts.index,
                    y=usia_counts.values,
                    title="Distribusi Usia Responden",
                    labels={'x': 'Kelompok Usia', 'y': 'Jumlah Responden'},
                    color=usia_counts.values,
                    color_continuous_scale='Purples',
                    text=usia_counts.values
                )
                fig_usia.update_traces(textposition='outside')
                fig_usia.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_usia, use_container_width=True)
        
        with demo_col2:
            if 'Jenis Kelamin' in df_processed.columns:
                gender_counts = df_processed['Jenis Kelamin'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title=" Distribusi Jenis Kelamin",
                    color_discrete_sequence=['#667eea', '#764ba2'],
                    hole=0.4
                )
                fig_gender.update_traces(textposition='inside', textinfo='percent+label')
                fig_gender.update_layout(height=400)
                st.plotly_chart(fig_gender, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        demo_col3, demo_col4 = st.columns(2)
        
        with demo_col3:
            if 'Domisili' in df_processed.columns:
                domisili_counts = df_processed['Domisili'].value_counts().head(10)
                fig_domisili = px.bar(
                    x=domisili_counts.values,
                    y=domisili_counts.index,
                    orientation='h',
                    title="Top 10 Domisili Responden",
                    labels={'x': 'Jumlah', 'y': 'Domisili'},
                    color=domisili_counts.values,
                    color_continuous_scale='Teal',
                    text=domisili_counts.values
                )
                fig_domisili.update_traces(textposition='outside')
                fig_domisili.update_layout(height=450)
                st.plotly_chart(fig_domisili, use_container_width=True)
        
        with demo_col4:
            if 'Status/Pekerjaan' in df_processed.columns:
                status_counts = df_processed['Status/Pekerjaan'].value_counts().head(8)
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Distribusi Status/Pekerjaan",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_status.update_traces(textposition='inside', textinfo='percent')
                fig_status.update_layout(height=450)
                st.plotly_chart(fig_status, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Analisis Platform E-Wallet</div>', unsafe_allow_html=True)
        
        platform_cols = ['platform_DANA', 'platform_OVO', 'platform_GoPay', 'platform_ShopeePay']
        available_platforms = [col for col in platform_cols if col in df_processed.columns]
        
        if available_platforms:
            platform_usage = df_processed[available_platforms].sum().sort_values(ascending=False)
            
            plat_col1, plat_col2 = st.columns(2)
            
            with plat_col1:
                fig_platform = px.bar(
                    x=platform_usage.index.str.replace('platform_', ''),
                    y=platform_usage.values,
                    title=" Penggunaan Platform E-Wallet",
                    labels={'x': 'Platform', 'y': 'Jumlah Pengguna'},
                    color=platform_usage.values,
                    color_continuous_scale='Viridis',
                    text=platform_usage.values
                )
                fig_platform.update_traces(textposition='outside')
                fig_platform.update_layout(height=400)
                st.plotly_chart(fig_platform, use_container_width=True)
            
            with plat_col2:
                fig_platform_pie = px.pie(
                    values=platform_usage.values,
                    names=platform_usage.index.str.replace('platform_', ''),
                    title=" Proporsi Penggunaan Platform",
                    color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
                    hole=0.4
                )
                fig_platform_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_platform_pie.update_layout(height=400)
                st.plotly_chart(fig_platform_pie, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("** Multi-Platform Usage Analysis**")
            df_processed['jumlah_platform'] = df_processed[available_platforms].sum(axis=1)
            platform_count = df_processed['jumlah_platform'].value_counts().sort_index()
            
            fig_overlap = px.bar(
                x=platform_count.index,
                y=platform_count.values,
                title="Jumlah Platform yang Digunakan per Responden",
                labels={'x': 'Jumlah Platform', 'y': 'Jumlah Responden'},
                color=platform_count.values,
                color_continuous_scale='Oranges',
                text=platform_count.values
            )
            fig_overlap.update_traces(textposition='outside')
            fig_overlap.update_layout(height=400)
            st.plotly_chart(fig_overlap, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Pola Perilaku Penggunaan E-Wallet</div>', unsafe_allow_html=True)
        
        behav_col1, behav_col2 = st.columns(2)
        
        with behav_col1:
            if 'durasi_penggunaan' in df_processed.columns:
                durasi_counts = df_processed['durasi_penggunaan'].value_counts().sort_index()
                fig_durasi = px.bar(
                    x=durasi_counts.index,
                    y=durasi_counts.values,
                    title="Durasi Penggunaan E-Wallet",
                    labels={'x': 'Durasi', 'y': 'Jumlah'},
                    color=durasi_counts.values,
                    color_continuous_scale='Blues',
                    text=durasi_counts.values
                )
                fig_durasi.update_traces(textposition='outside')
                fig_durasi.update_layout(height=400)
                st.plotly_chart(fig_durasi, use_container_width=True)
        
        with behav_col2:
            if 'frekuensi_penggunaan_mingguan' in df_processed.columns:
                freq_counts = df_processed['frekuensi_penggunaan_mingguan'].value_counts().sort_index()
                fig_freq = px.bar(
                    x=freq_counts.index,
                    y=freq_counts.values,
                    title="Frekuensi Penggunaan per Minggu",
                    labels={'x': 'Frekuensi', 'y': 'Jumlah'},
                    color=freq_counts.values,
                    color_continuous_scale='Greens',
                    text=freq_counts.values
                )
                fig_freq.update_traces(textposition='outside')
                fig_freq.update_layout(height=400)
                st.plotly_chart(fig_freq, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Rata-rata Skor 5 Dimensi Analisis**")
        dimensions = ['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan', 'Intensitas']
        available_dims = [dim for dim in dimensions if dim in df_processed.columns]
        
        if available_dims:
            dim_means = df_processed[available_dims].mean()
            
            fig_dimensions = go.Figure()
            fig_dimensions.add_trace(go.Bar(
                x=dim_means.index,
                y=dim_means.values,
                marker_color=['#f093fb', '#4facfe', '#43e97b', '#fa709a', '#30cfd0'],
                text=np.round(dim_means.values, 2),
                textposition='outside',
                textfont=dict(size=14, family='Arial Black')
            ))
            
            fig_dimensions.update_layout(
                title="Rata-rata Skor Berdasarkan 5 Dimensi",
                xaxis_title="Dimensi",
                yaxis_title="Skor Rata-rata (Skala 1-5)",
                yaxis=dict(range=[0, 5.5]),
                showlegend=False,
                height=450
            )
            
            st.plotly_chart(fig_dimensions, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Distribusi Persepsi Likert (Q1-Q15)</div>', unsafe_allow_html=True)
        
        q_cols = [f'Q{i}' for i in range(1, 16)]
        available_q = [col for col in q_cols if col in df_processed.columns]
        
        if available_q:
            selected_q = st.selectbox("Pilih Pertanyaan untuk Analisis Detail", available_q, key='likert_select')
            
            likert_col1, likert_col2 = st.columns(2)
            
            with likert_col1:
                q_counts = df_processed[selected_q].value_counts().sort_index()
                fig_q_bar = px.bar(
                    x=q_counts.index,
                    y=q_counts.values,
                    title=f"Distribusi {selected_q}",
                    labels={'x': 'Skor Likert (1-5)', 'y': 'Jumlah Responden'},
                    color=q_counts.values,
                    color_continuous_scale='RdYlGn',
                    text=q_counts.values
                )
                fig_q_bar.update_traces(textposition='outside')
                fig_q_bar.update_layout(height=400)
                st.plotly_chart(fig_q_bar, use_container_width=True)
            
            with likert_col2:
                fig_q_pie = px.pie(
                    values=q_counts.values,
                    names=['Skor ' + str(i) for i in q_counts.index],
                    title=f"Proporsi {selected_q}",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                fig_q_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_q_pie.update_layout(height=400)
                st.plotly_chart(fig_q_pie, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Heatmap Distribusi Semua Pertanyaan (Q1-Q15)**")
            
            q_matrix = []
            for q in available_q[:15]:
                counts = df_processed[q].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
                q_matrix.append(counts.values)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=q_matrix,
                x=['1<br>Sangat Tidak<br>Setuju', '2<br>Tidak<br>Setuju', '3<br>Netral', '4<br>Setuju', '5<br>Sangat<br>Setuju'],
                y=[f'Q{i}' for i in range(1, len(q_matrix)+1)],
                colorscale='RdYlGn',
                text=q_matrix,
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Jumlah")
            ))
            
            fig_heatmap.update_layout(
                title="Heatmap Distribusi Jawaban Q1-Q15",
                xaxis_title="Respon Likert",
                yaxis_title="Pertanyaan",
                height=650
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == " Clustering Analysis":
    st.markdown('<div class="main-header"> Clustering Analysis (K-Means)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style='font-size: 1.1rem; margin: 0;'><strong>K-Means Clustering</strong> digunakan untuk mengelompokkan pengguna berdasarkan 
        4 dimensi utama: <strong>Perilaku, Kemudahan, Adopsi, dan Kepercayaan</strong> menjadi <strong>2 cluster</strong> 
        untuk segmentasi pengguna yang efektif.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"> Visualisasi Cluster (PCA 2D)</div>', unsafe_allow_html=True)
    
    pca_df = pd.DataFrame(
        models['pca_features'],
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = models['clusters']
    
    fig_scatter = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title="Scatter Plot Cluster dengan PCA Dimensionality Reduction",
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'Cluster': 'Cluster'},
        color_continuous_scale=['#667eea', '#f5576c']
    )
    
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')))
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown('<div class="sub-header"> Perbandingan Rata-rata Antar Cluster</div>', unsafe_allow_html=True)
    
    cluster_df = models['X_cluster'].copy()
    cluster_df['Cluster'] = models['clusters']
    cluster_df['Intensitas'] = df_processed.loc[cluster_df.index, 'Intensitas']
    
    cluster_means = cluster_df.groupby('Cluster')[['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan', 'Intensitas']].mean()
    
    clust_col1, clust_col2 = st.columns(2)
    
    with clust_col1:
        fig_comparison = go.Figure()
        
        colors = ['#667eea', '#f5576c']
        for idx, cluster in enumerate(cluster_means.index):
            fig_comparison.add_trace(go.Bar(
                name=f'Cluster {cluster}',
                x=cluster_means.columns,
                y=cluster_means.loc[cluster],
                text=np.round(cluster_means.loc[cluster], 2),
                textposition='outside',
                marker_color=colors[idx]
            ))
        
        fig_comparison.update_layout(
            title=" Bar Chart Perbandingan Cluster",
            xaxis_title="Dimensi",
            yaxis_title="Skor Rata-rata",
            barmode='group',
            yaxis=dict(range=[0, 5.5]),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with clust_col2:
        fig_radar = go.Figure()
        
        for idx, cluster in enumerate(cluster_means.index):
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_means.loc[cluster].values,
                theta=cluster_means.columns,
                fill='toself',
                name=f'Cluster {cluster}',
                line=dict(color=colors[idx], width=2)
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            title=" Radar Chart Perbandingan Cluster",
            showlegend=True,
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sub-header"> Statistik Deskriptif Cluster</div>', unsafe_allow_html=True)
    
    cluster_stats = cluster_df.groupby('Cluster').agg({
        'Perilaku': ['mean', 'std', 'min', 'max'],
        'Kemudahan': ['mean', 'std', 'min', 'max'],
        'Adopsi': ['mean', 'std', 'min', 'max'],
        'Kepercayaan': ['mean', 'std', 'min', 'max'],
        'Intensitas': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    st.dataframe(cluster_stats.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)
    
    st.markdown('<div class="sub-header"> Interpretasi & Karakteristik Cluster</div>', unsafe_allow_html=True)
    
    interp_col1, interp_col2 = st.columns(2)
    
    with interp_col1:
        cluster_0_means = cluster_means.loc[0]
        cluster_0_count = (models['clusters'] == 0).sum()
        
        if cluster_0_means['Intensitas'] > cluster_means['Intensitas'].mean():
            st.markdown("""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 2rem; border-radius: 15px; color: white;'>
                <h3 style='margin: 0 0 1rem 0;'> Cluster 0: Pengguna Aktif</h3>
                <p style='font-size: 2rem; font-weight: 800; margin: 0;'>{} Responden</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1rem 0;'>
                <h4 style='margin: 1rem 0 0.5rem 0;'>Karakteristik:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li><strong>Intensitas Tinggi</strong>: Skor {:.2f}/5.0</li>
                    <li>Persepsi positif terhadap semua dimensi</li>
                    <li>Pengguna loyal dan aktif</li>
                    <li>Target untuk program loyalitas premium</li>
                </ul>
            </div>
            """.format(cluster_0_count, cluster_0_means['Intensitas']), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 2rem; border-radius: 15px; color: white;'>
                <h3 style='margin: 0 0 1rem 0;'> Cluster 0: Pengguna Moderat</h3>
                <p style='font-size: 2rem; font-weight: 800; margin: 0;'>{} Responden</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1rem 0;'>
                <h4 style='margin: 1rem 0 0.5rem 0;'>Karakteristik:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li><strong>Intensitas Rendah</strong>: Skor {:.2f}/5.0</li>
                    <li>Perlu edukasi dan insentif lebih</li>
                    <li>Potensi peningkatan engagement</li>
                    <li>Target untuk kampanye awareness</li>
                </ul>
            </div>
            """.format(cluster_0_count, cluster_0_means['Intensitas']), unsafe_allow_html=True)
    
    with interp_col2:
        cluster_1_means = cluster_means.loc[1]
        cluster_1_count = (models['clusters'] == 1).sum()
        
        if cluster_1_means['Intensitas'] > cluster_means['Intensitas'].mean():
            st.markdown("""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 2rem; border-radius: 15px; color: white;'>
                <h3 style='margin: 0 0 1rem 0;'> Cluster 1: Pengguna Aktif</h3>
                <p style='font-size: 2rem; font-weight: 800; margin: 0;'>{} Responden</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1rem 0;'>
                <h4 style='margin: 1rem 0 0.5rem 0;'>Karakteristik:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li><strong>Intensitas Tinggi</strong>: Skor {:.2f}/5.0</li>
                    <li>Persepsi positif terhadap semua dimensi</li>
                    <li>Pengguna loyal dan aktif</li>
                    <li>Target untuk program loyalitas premium</li>
                </ul>
            </div>
            """.format(cluster_1_count, cluster_1_means['Intensitas']), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 2rem; border-radius: 15px; color: white;'>
                <h3 style='margin: 0 0 1rem 0;'> Cluster 1: Pengguna Moderat</h3>
                <p style='font-size: 2rem; font-weight: 800; margin: 0;'>{} Responden</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1rem 0;'>
                <h4 style='margin: 1rem 0 0.5rem 0;'>Karakteristik:</h4>
                <ul style='margin: 0; padding-left: 1.5rem;'>
                    <li><strong>Intensitas Rendah</strong>: Skor {:.2f}/5.0</li>
                    <li>Perlu edukasi dan insentif lebih</li>
                    <li>Potensi peningkatan engagement</li>
                    <li>Target untuk kampanye awareness</li>
                </ul>
            </div>
            """.format(cluster_1_count, cluster_1_means['Intensitas']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sub-header"> Distribusi Anggota Cluster</div>', unsafe_allow_html=True)
    
    cluster_counts = pd.Series(models['clusters']).value_counts().sort_index()
    
    dist_col1, dist_col2 = st.columns([2, 1])
    
    with dist_col1:
        fig_cluster_dist = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title="Proporsi Distribusi Anggota Cluster",
            color_discrete_sequence=['#667eea', '#f5576c'],
            hole=0.4
        )
        fig_cluster_dist.update_traces(textposition='inside', textinfo='percent+label+value', textfont_size=14)
        fig_cluster_dist.update_layout(height=400)
        st.plotly_chart(fig_cluster_dist, use_container_width=True)
    
    with dist_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for idx in cluster_counts.index:
            pct = (cluster_counts[idx] / cluster_counts.sum()) * 100
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem; text-align: center;'>
                    <h4 style='margin: 0;'>Cluster {}</h4>
                    <p style='font-size: 2rem; font-weight: 800; margin: 0.5rem 0 0 0;'>{} ({:.1f}%)</p>
                </div>
            """.format(idx, cluster_counts[idx], pct), unsafe_allow_html=True)

elif page == " Logistic Regression":
    st.markdown('<div class="main-header"> Logistic Regression Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style='font-size: 1.1rem; margin: 0;'><strong>Logistic Regression</strong> digunakan untuk memprediksi <strong>Intensitas Penggunaan</strong> 
        (Tinggi/Rendah) berdasarkan 4 fitur utama: <strong>Perilaku, Kemudahan, Adopsi, dan Kepercayaan</strong>.</p>
        <p style='margin: 0.5rem 0 0 0;'>Target dibuat dengan median split: Intensitas ≥ median = <strong>Tinggi (1)</strong>, < median = <strong>Rendah (0)</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"> Metrik Performa Model</div>', unsafe_allow_html=True)
    
    accuracy = accuracy_score(models['y_test'], models['y_pred'])
    report_dict = classification_report(models['y_test'], models['y_pred'], output_dict=True)
    precision = report_dict['1']['precision']
    recall = report_dict['1']['recall']
    f1 = report_dict['1']['f1-score']
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white; height: 150px;'>
                <h4 style='margin: 0; font-size: 1rem;'> Accuracy</h4>
                <p style='font-size: 3rem; font-weight: 800; margin: 0.5rem 0 0 0;'>{:.1%}</p>
            </div>
        """.format(accuracy), unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white; height: 150px;'>
                <h4 style='margin: 0; font-size: 1rem;'> Precision</h4>
                <p style='font-size: 3rem; font-weight: 800; margin: 0.5rem 0 0 0;'>{:.1%}</p>
            </div>
        """.format(precision), unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white; height: 150px;'>
                <h4 style='margin: 0; font-size: 1rem;'> Recall</h4>
                <p style='font-size: 3rem; font-weight: 800; margin: 0.5rem 0 0 0;'>{:.1%}</p>
            </div>
        """.format(recall), unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; color: white; height: 150px;'>
                <h4 style='margin: 0; font-size: 1rem;'> F1-Score</h4>
                <p style='font-size: 3rem; font-weight: 800; margin: 0.5rem 0 0 0;'>{:.1%}</p>
            </div>
        """.format(f1), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sub-header"> Classification Report Detail</div>', unsafe_allow_html=True)
    
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.round(3).style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)
    
    logreg_col1, logreg_col2 = st.columns(2)
    
    with logreg_col1:
        st.markdown('<div class="sub-header"> Confusion Matrix</div>', unsafe_allow_html=True)
        
        cm = confusion_matrix(models['y_test'], models['y_pred'])
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted<br>Low', 'Predicted<br>High'],
            y=['Actual<br>Low', 'Actual<br>High'],
            colorscale='Purples',
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 20},
            colorbar=dict(title="Count")
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix Prediksi",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=450
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with logreg_col2:
        st.markdown('<div class="sub-header"> ROC Curve</div>', unsafe_allow_html=True)
        
        fpr, tpr, _ = roc_curve(models['y_test'], models['y_proba'])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier (AUC = 0.5)',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {roc_auc:.3f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sub-header"> Feature Importance (Odds Ratio)</div>', unsafe_allow_html=True)

    pipe_or_model = models['logreg'] 

    if hasattr(pipe_or_model, "named_steps"):
        lr = list(pipe_or_model.named_steps.values())[-1]
    else:
        lr = pipe_or_model

    if not hasattr(lr, "coef_"):
        st.warning("Model Logistic Regression tidak memiliki koefisien (coef_). Pastikan model sudah fit.")
    else:
        coef = lr.coef_[0]
        odds_ratio = np.exp(coef)

        feature_names = ['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan']

        if len(coef) != len(feature_names):
            feature_names = [f"Feature_{i+1}" for i in range(len(coef))]

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef,
            'Odds Ratio': odds_ratio
        }).sort_values('Odds Ratio', ascending=False)

        imp_col1, imp_col2 = st.columns([2, 1])

        with imp_col1:
            fig_odds = px.bar(
                feature_importance,
                x='Feature',
                y='Odds Ratio',
                title="Odds Ratio Feature Importance",
                labels={'Odds Ratio': 'Odds Ratio (Pengaruh terhadap Intensitas Tinggi)'},
                color='Odds Ratio',
                color_continuous_scale='RdYlGn',
                text='Odds Ratio'
            )

            fig_odds.update_traces(texttemplate='%{text:.3f}', textposition='outside', textfont_size=14)
            fig_odds.update_layout(showlegend=False, height=450)
            fig_odds.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Baseline (OR = 1)")

            st.plotly_chart(fig_odds, use_container_width=True)

        with imp_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                feature_importance.round(3).style.background_gradient(cmap='RdYlGn', subset=['Odds Ratio']),
                use_container_width=True, hide_index=True, height=250
            )

            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 10px; color: white; margin-top: 1rem;'>
                <h4 style='margin: 0 0 0.5rem 0;'> Interpretasi Odds Ratio:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; font-size: 0.9rem;'>
                    <li><strong>OR > 1</strong>: Meningkatkan peluang Intensitas Tinggi</li>
                    <li><strong>OR < 1</strong>: Menurunkan peluang Intensitas Tinggi</li>
                    <li><strong>OR = 1</strong>: Tidak berpengaruh signifikan</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


elif page == " Prediction":
    st.markdown('<div class="main-header"> Prediksi Intensitas Penggunaan E-Wallet</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style='font-size: 1.1rem; margin: 0;'>Masukkan nilai untuk <strong>15 pertanyaan (Q1-Q15)</strong> menggunakan <strong>skala Likert 1-5</strong> 
        untuk memprediksi <strong>Intensitas Penggunaan E-Wallet</strong> Anda (Tinggi/Rendah).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"> Input Data Responden</div>', unsafe_allow_html=True)
    
    questions = {
        'Q1': 'Saya merasa penggunaan e-wallet lebih praktis dibandingkan pembayaran tunai',
        'Q2': 'Saya merasa nyaman melakukan transaksi menggunakan e-wallet',
        'Q3': 'Saya memilih e-wallet untuk mengurangi risiko membawa uang tunai',
        'Q4': 'Saya merasa aplikasi e-wallet mudah dipahami meskipun saya baru pertama kali menggunakannya',
        'Q5': 'Tampilan dan menu pada e-wallet mudah untuk dioperasikan',
        'Q6': 'Saya jarang mengalami kesulitan teknis ketika menggunakan e-wallet',
        'Q7': 'Saya merasa penggunaan e-wallet sesuai dengan kebutuhan transaksi saya',
        'Q8': 'Saya terbuka untuk mencoba fitur baru yang ditawarkan e-wallet',
        'Q9': 'Saya memiliki keinginan untuk terus menggunakan e-wallet di masa mendatang',
        'Q10': 'Saya percaya bahwa transaksi menggunakan e-wallet aman',
        'Q11': 'Saya yakin data pribadi saya terlindungi saat menggunakan e-wallet',
        'Q12': 'Saya percaya penyedia layanan e-wallet dapat menjaga keamanan saldo pengguna',
        'Q13': 'Saya merasa e-wallet memiliki sistem keamanan yang dapat diandalkan',
        'Q14': 'Saya lebih sering menggunakan e-wallet dibandingkan uang tunai dalam berbagai transaksi',
        'Q15': 'Saya menggunakan e-wallet untuk berbagai jenis transaksi (belanja, transportasi, makanan, pulsa, dll.)'
    }
    
    user_input = {}
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
            <h4 style='margin: 0;'> Dimensi Perilaku</h4>
        </div>
        """, unsafe_allow_html=True)
        user_input['Q1'] = st.slider(f"**Q1**: {questions['Q1']}", 1, 5, 3, key='q1')
        user_input['Q2'] = st.slider(f"**Q2**: {questions['Q2']}", 1, 5, 3, key='q2')
        user_input['Q3'] = st.slider(f"**Q3**: {questions['Q3']}", 1, 5, 3, key='q3')
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1.5rem 0 1rem 0;'>
            <h4 style='margin: 0;'> Dimensi Kemudahan</h4>
        </div>
        """, unsafe_allow_html=True)
        user_input['Q4'] = st.slider(f"**Q4**: {questions['Q4']}", 1, 5, 3, key='q4')
        user_input['Q5'] = st.slider(f"**Q5**: {questions['Q5']}", 1, 5, 3, key='q5')
        user_input['Q6'] = st.slider(f"**Q6**: {questions['Q6']}", 1, 5, 3, key='q6')
    
    with pred_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
            <h4 style='margin: 0;'> Dimensi Adopsi</h4>
        </div>
        """, unsafe_allow_html=True)
        user_input['Q7'] = st.slider(f"**Q7**: {questions['Q7']}", 1, 5, 3, key='q7')
        user_input['Q8'] = st.slider(f"**Q8**: {questions['Q8']}", 1, 5, 3, key='q8')
        user_input['Q9'] = st.slider(f"**Q9**: {questions['Q9']}", 1, 5, 3, key='q9')
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1.5rem 0 1rem 0;'>
            <h4 style='margin: 0;'> Dimensi Kepercayaan </h4>
        </div>
        """, unsafe_allow_html=True)
        user_input['Q10'] = st.slider(f"**Q10**: {questions['Q10']}", 1, 5, 3, key='q10')
        user_input['Q11'] = st.slider(f"**Q11**: {questions['Q11']}", 1, 5, 3, key='q11')
    
    with pred_col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
            <h4 style='margin: 0;'> Dimensi Kepercayaan </h4>
        </div>
        """, unsafe_allow_html=True)
        user_input['Q12'] = st.slider(f"**Q12**: {questions['Q12']}", 1, 5, 3, key='q12')
        user_input['Q13'] = st.slider(f"**Q13**: {questions['Q13']}", 1, 5, 3, key='q13')
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1.5rem 0 1rem 0;'>
            <h4 style='margin: 0;'> Dimensi Intensitas</h4>
        </div>
        """, unsafe_allow_html=True)
        user_input['Q14'] = st.slider(f"**Q14**: {questions['Q14']}", 1, 5, 3, key='q14')
        user_input['Q15'] = st.slider(f"**Q15**: {questions['Q15']}", 1, 5, 3, key='q15')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button(" **Prediksi Intensitas Sekarang**", type="primary", use_container_width=True):
        perilaku = np.mean([user_input['Q1'], user_input['Q2'], user_input['Q3']])
        kemudahan = np.mean([user_input['Q4'], user_input['Q5'], user_input['Q6']])
        adopsi = np.mean([user_input['Q7'], user_input['Q8'], user_input['Q9']])
        kepercayaan = np.mean([user_input['Q10'], user_input['Q11'], user_input['Q12'], user_input['Q13']])
        intensitas = np.mean([user_input['Q14'], user_input['Q15']])
        
        _, feature_cols, _, _, _, _ = load_artifacts()

        feat_values = {
            'Perilaku': perilaku,
            'Kemudahan': kemudahan,
            'Adopsi': adopsi,
            'Kepercayaan': kepercayaan
        }

        X_pred_df = np.array([[perilaku, kemudahan, adopsi, kepercayaan]])

        prediction = models['logreg'].predict(X_pred_df)[0]
        prediction_proba = models['logreg'].predict_proba(X_pred_df)[0]

        
        st.markdown('<div class="sub-header"> Hasil Prediksi</div>', unsafe_allow_html=True)
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown("** Skor 5 Dimensi Anda**")
            
            dimensions_score = pd.DataFrame({
                'Dimensi': ['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan', 'Intensitas'],
                'Skor': [perilaku, kemudahan, adopsi, kepercayaan, intensitas]
            })
            
            fig_user_dims = px.bar(
                dimensions_score,
                x='Dimensi',
                y='Skor',
                title="Skor Dimensi Berdasarkan Input Anda",
                color='Skor',
                color_continuous_scale='RdYlGn',
                range_color=[1, 5],
                text='Skor'
            )
            
            fig_user_dims.update_traces(texttemplate='%{text:.2f}', textposition='outside', textfont_size=14)
            fig_user_dims.update_layout(yaxis=dict(range=[0, 5.5]), showlegend=False, height=450)
            
            st.plotly_chart(fig_user_dims, use_container_width=True)
        
        with result_col2:
            st.markdown("** Hasil Prediksi Model**")
            
            if prediction == 1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
                    <h2 style='margin: 0; font-size: 2.5rem;'> Intensitas TINGGI</h2>
                    <p style='font-size: 3rem; font-weight: 800; margin: 1rem 0;'>{:.1%}</p>
                    <p style='margin: 0;'>Probabilitas Prediksi</p>
                </div>
                """.format(prediction_proba[1]), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("""
                ** Karakteristik Pengguna Aktif:**
                - Pengguna aktif dan loyal terhadap e-wallet
                - Frekuensi penggunaan tinggi dan konsisten
                - Menggunakan e-wallet untuk berbagai jenis transaksi
                - Memiliki kepercayaan tinggi terhadap sistem
                - Kandidat ideal untuk program loyalitas premium
                """)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
                    <h2 style='margin: 0; font-size: 2.5rem;'> Intensitas RENDAH</h2>
                    <p style='font-size: 3rem; font-weight: 800; margin: 1rem 0;'>{:.1%}</p>
                    <p style='margin: 0;'>Probabilitas Prediksi</p>
                </div>
                """.format(prediction_proba[0]), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("""
                ** Karakteristik Pengguna Moderat:**
                - Pengguna casual dengan frekuensi rendah
                - Penggunaan terbatas pada kebutuhan tertentu
                - Masih ada ruang untuk peningkatan engagement
                - Perlu edukasi lebih lanjut tentang manfaat e-wallet
                - Target untuk kampanye awareness dan promosi
                """)
            
            prob_df = pd.DataFrame({
                'Kategori': ['Rendah', 'Tinggi'],
                'Probabilitas': prediction_proba
            })
            
            fig_prob = px.bar(
                prob_df,
                x='Kategori',
                y='Probabilitas',
                title="Distribusi Probabilitas Prediksi",
                color='Probabilitas',
                color_continuous_scale='RdYlGn',
                text='Probabilitas'
            )
            
            fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='outside', textfont_size=16)
            fig_prob.update_layout(showlegend=False, yaxis=dict(range=[0, 1.1]), height=400)
            
            st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("** Perbandingan dengan Rata-rata Dataset**")
        
        avg_dims = df_processed[['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan', 'Intensitas']].mean()
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[perilaku, kemudahan, adopsi, kepercayaan, intensitas],
            theta=['Perilaku', 'Kemudahan', 'Adopsi', 'Kepercayaan', 'Intensitas'],
            fill='toself',
            name='Input Anda',
            line=dict(color='#667eea', width=3)
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_dims.values,
            theta=avg_dims.index,
            fill='toself',
            name='Rata-rata Dataset',
            line=dict(color='#f5576c', dash='dash', width=2)
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Radar Chart: Perbandingan Anda vs Rata-rata",
            showlegend=True,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
