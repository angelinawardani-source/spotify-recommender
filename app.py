import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Spotify Pro Recommender", layout="wide")

# CSS Custom: BASE PUTIH + PANEL & KOTAK GRADASI UNGU + FIX TOTAL JALUR SLIDER GLOBAL
st.markdown("""
    <style>
    /* =========================================================================
       MENEMBAK VARIABEL UTAMA STREAMLIT (Memaksa warna dasar biru menjadi ungu)
       ========================================================================= */
    :root {
        --primary-color: #5e35b1 !important; /* Mengubah warna utama slider & komponen aktif */
    }
    
    html, body, [class*="css"], .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #FFFFFF !important;
        color: #222222 !important;
    }
    
    /* =========================================================================
       SIDEBAR KIRI: GRADASI UNGU LEMBUT
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #7e57c2 0%, #b39ddb 50%, #d1c4e9 100%) !important;
    }
    
    /* Teks judul utama di dalam sidebar (Area Ungu Atas) */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Teks biasa di dalam sidebar (Area Ungu Bawah) menggunakan warna Ungu Tua */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #3b0975 !important;
        font-weight: 600 !important;
    }
    
    /* Warna teks angka slider di bawah agar tetap kontras */
    section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
        color: #3b0975 !important;
        font-weight: bold !important;
    }
    
    /* =========================================================================
       ANTISIPASI CSS TAMBAHAN UNTUK SLIDER DI SIDEBAR
       ========================================================================= */
    /* Mengubah warna latar belakang bar jalur aktif menjadi ungu muda pastel */
    div[data-testid="stSidebar"] div[data-testid="stSlider"] [data-track="true"] > div > div,
    div[data-testid="stSidebar"] .stSlider [data-track="true"] > div div {
        background: #b39ddb !important;
        background-color: #b39ddb !important;
    }
    
    /* Mengubah pegangan bulatan menjadi ungu tua solid */
    div[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] {
        background-color: #5e35b1 !important;
        border-color: #5e35b1 !important;
    }
    
    /* =========================================================================
       KOTAK REKOMENDASI: GRADASI HALUS DENGAN UJUNG BAWAH UNGU LILAC PASTEL
       ========================================================================= */
    .mini-song-card {
        background: linear-gradient(to bottom, #7e57c2 0%, #b39ddb 50%, #d1c4e9 100%) !important;
        border-radius: 14px !important; 
        padding: 15px !important;
        box-shadow: 0 6px 20px rgba(126, 87, 194, 0.15) !important;
        
        /* Ukuran kotak dikunci seragam */
        min-width: 220px !important; 
        max-width: 220px !important;
        height: 240px !important; 
        
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        flex-shrink: 0 !important;
    }
    
    /* Area Ungu Atas: Teks Judul Lagu & Artis */
    .mini-song-card b {
        color: #FFFFFF !important; 
        font-size: 0.95rem !important;
        white-space: normal !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }
    .mini-song-card .artist-text {
        color: #F3E5F5 !important; 
        font-size: 0.85rem !important;
        font-weight: bold !important;
        display: block !important;
        margin-top: 4px !important;
    }
    
    /* Area Ungu Bawah: Teks rincian lagu dibuat Ungu Tua Pekat agar kontras */
    .mini-song-card .stat-text {
        color: #3b0975 !important; 
        font-size: 0.75rem !important;
        font-weight: 600 !important;
    }
    
    /* =========================================================================
       ELEMEN HALAMAN UTAMA (Wadah Scroll & Judul Halaman)
       ========================================================================= */
    h1, h2, h3, h4, h5, h6 {
        color: #7e57c2 !important;
    }
    
    .scroll-container {
        display: flex !important;
        overflow-x: auto !important;
        gap: 16px !important;
        padding: 10px 5px 25px 5px !important;
        width: 100% !important;
    }
    
    .scroll-container::-webkit-scrollbar {
        height: 8px;
    }
    .scroll-container::-webkit-scrollbar-track {
        background: #F5F5F5;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background: #7e57c2;
        border-radius: 4px;
    }
    
    .card-divider {
        border-top: 1px solid rgba(59, 9, 117, 0.15) !important;
        margin-top: 5px !important;
        padding-top: 5px !important;
    }
    
    hr {
        border-color: #E0E0E0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_clustered.csv")
    return df.sort_values('track_name')

df = load_data()

# ==========================================
# A. SIDEBAR
# ==========================================
st.sidebar.header("Control Panel")
st.sidebar.write("Tentukan lagu favoritmu dan atur jumlah rekomendasi.")

pilihan_lagu = st.sidebar.selectbox(
    "Pilih Lagu yang Kamu Suka:",
    df['track_name'].unique()
)

jumlah_rek = st.sidebar.slider("Jumlah Rekomendasi:", 5, 15, 10)

st.sidebar.markdown("---")
st.sidebar.write("Powered by K-Means Clustering")

# ==========================================
# B. HALAMAN UTAMA
# ==========================================
st.title("🎵 Spotify Music Recommendation System")
st.write(f"Menampilkan rekomendasi berdasarkan kemiripan karakter suara dengan lagu **{pilihan_lagu}**")

data_lagu_pilihan = df[df['track_name'] == pilihan_lagu].iloc[0]
target_cluster = data_lagu_pilihan['cluster']
nama_cluster = data_lagu_pilihan['cluster_name']

rekomendasi_df = df[(df['cluster'] == target_cluster) & (df['track_name'] != pilihan_lagu)] \
                .sort_values(by='popularity', ascending=False) \
                .head(jumlah_rek)

st.info(f"Lagu ini masuk dalam kategori: **{nama_cluster}** (Cluster {target_cluster})")
st.subheader("Daftar Rekomendasi Untukmu")

html_cards = ""
for index, row in rekomendasi_df.iterrows():
    html_cards += (
        '<div class="mini-song-card">'
            '<div>'
                '<b style="line-height: 1.2;">' + str(row['track_name']) + '</b>'
                '<span class="artist-text">👤 ' + str(row['artists']) + '</span>'
            '</div>'
            '<div>'
                '<span class="stat-text" style="display: block; margin-bottom: 2px;">Popularity: ' + str(row['popularity']) + '</span>'
                '<p class="card-divider stat-text" style="margin: 0; line-height: 1.3;">'
                    '&bull; Dance: ' + str(row['danceability']) + '<br>'
                    '&bull; Energy: ' + str(row['energy']) + '<br>'
                    '&bull; Acoustic: ' + str(row['acousticness']) + '<br>'
                    '&bull; Valence: ' + str(row['valence']) +
                '</p>'
            '</div>'
        '</div>'
    )

st.markdown('<div class="scroll-container">' + html_cards + '</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# C. GRAFIK TOP ARTISTS
# ==========================================
st.subheader(f"Top 5 Artis di Kategori {nama_cluster}")
st.write("Musisi yang paling mendominasi karakter suara di cluster ini.")

top_artists = df[df['cluster'] == target_cluster]['artists'].value_counts().head(5).reset_index()
top_artists.columns = ['Artist', 'Jumlah Lagu']

# Membalik data agar bar tertinggi berada di posisi paling atas grafik
top_artists = top_artists.iloc[::-1].reset_index(drop=True)

# Rentang 5 warna gradasi melompat tegas dari ungu muda ke tua agar tidak samar
custom_purples = ['#d1c4e9', '#b39ddb', '#9575cd', '#7e57c2', '#5e35b1']

fig = px.bar(
    top_artists, 
    x='Jumlah Lagu', 
    y='Artist', 
    orientation='h',
    color=top_artists.index, 
    color_continuous_scale=custom_purples
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False, 
    height=350, 
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(tickfont=dict(color="#222222"), showgrid=False),
    yaxis=dict(tickfont=dict(color="#222222"), showgrid=False)
)

# Sembunyikan garis skala warna kanan agar minimalis
fig.update_coloraxes(showscale=False)

st.plotly_chart(fig, use_container_width=True)
st.caption("Data ditampilkan berdasarkan analisis Machine Learning terhadap dataset Spotify.")