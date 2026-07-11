import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Spotify Pro Recommender", layout="wide")

# CSS Custom: Pengaturan tema visual aplikasi web
st.markdown("""
    <style>
    :root {
        --primary-color: #5e35b1 !important;
    }
    
    html, body, [class*="css"], .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #FFFFFF !important;
        color: #222222 !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #7e57c2 0%, #b39ddb 50%, #d1c4e9 100%) !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #3b0975 !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
        color: #3b0975 !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stSidebar"] div[data-testid="stSlider"] [data-track="true"] > div > div,
    div[data-testid="stSidebar"] .stSlider [data-track="true"] > div div {
        background: #b39ddb !important;
        background-color: #b39ddb !important;
    }
    
    div[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] {
        background-color: #5e35b1 !important;
        border-color: #5e35b1 !important;
    }
    
    .mini-song-card {
        background: linear-gradient(to bottom, #7e57c2 0%, #b39ddb 50%, #d1c4e9 100%) !important;
        border-radius: 14px !important; 
        padding: 15px !important;
        box-shadow: 0 6px 20px rgba(126, 87, 194, 0.15) !important;
        min-width: 220px !important; 
        max-width: 220px !important;
        height: 240px !important; 
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        flex-shrink: 0 !important;
    }
    
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
    
    .mini-song-card .stat-text {
        color: #3b0975 !important; 
        font-size: 0.75rem !important;
        font-weight: 600 !important;
    }
    
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

# 2. Memuat Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_clustered.csv")
    return df.sort_values('track_name')

df = load_data()

# 3. Sidebar Panel Kontrol
st.sidebar.header("Control Panel")


pilihan_lagu = st.sidebar.selectbox(
    "Cari Lagu yang Kamu Suka:",
    df['track_name'].unique()
)

jumlah_rek = st.sidebar.slider("Jumlah Rekomendasi:", 5, 15, 10)

# Menambahkan tombol eksekusi pencarian di sidebar
tombol_cari = st.sidebar.button("Cari Rekomendasi", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Powered by Hybrid Recommendation (K-Means + Cosine Similarity)")

# 4. Pemrosesan Halaman Utama
st.title("🎵 Spotify Music Recommendation System")

# Seluruh logika dijalankan hanya jika tombol ditekan
if tombol_cari:
    st.write(f"Menampilkan rekomendasi berdasarkan kemiripan karakter suara dengan lagu **{pilihan_lagu}**")

    # Ekstraksi informasi data lagu pilihan pengguna
    data_lagu_pilihan = df[df['track_name'] == pilihan_lagu].iloc[0]
    target_cluster = data_lagu_pilihan['cluster']
    nama_cluster = data_lagu_pilihan['cluster_name']

    # Definisi fitur audio untuk kalkulasi jarak kemiripan geometri
    fitur_audio = ['danceability', 'energy', 'acousticness', 'valence']

    # Implementasi Logika Sistem Rekomendasi Hybrid
    # Tahap 1: Filtrasi awal berdasarkan kelompok klaster K-Means yang serupa
    cluster_df = df[df['cluster'] == target_cluster].reset_index(drop=True)

    # Tahap 2: Filtrasi berdasarkan nilai popularitas dengan batas minimum 50
    top_popular_in_cluster = cluster_df[cluster_df['popularity'] >= 50].reset_index(drop=True)

    # Antisipasi sistem: Jika tidak ditemukan lagu dengan popularitas >= 50, maka digunakan ambang batas 100 lagu terpopuler
    if top_popular_in_cluster.empty:
        top_popular_in_cluster = cluster_df.sort_values(by='popularity', ascending=False).head(100).reset_index(drop=True)

    # Validasi inklusi lagu pilihan pengguna di dalam matriks komparasi
    if pilihan_lagu.lower() not in top_popular_in_cluster['track_name'].str.lower().values:
        top_popular_in_cluster = pd.concat([df[df['track_name'] == pilihan_lagu], top_popular_in_cluster]).reset_index(drop=True)

    # Tahap 3: Perhitungan nilai kedekatan fitur menggunakan matriks Cosine Similarity
    fitur_lagu_pilihan = df[df['track_name'] == pilihan_lagu][fitur_audio]
    fitur_kandidat_klaster = top_popular_in_cluster[fitur_audio]

    skor_kemiripan = cosine_similarity(fitur_lagu_pilihan, fitur_kandidat_klaster)[0]
    top_popular_in_cluster['similarity_score'] = skor_kemiripan

    # Tahap 4: Pengurutan hasil berdasarkan skor kemiripan tertinggi dan mereduksi duplikasi input awal
    rekomendasi_df = top_popular_in_cluster[top_popular_in_cluster['track_name'] != pilihan_lagu]
    rekomendasi_df = rekomendasi_df.sort_values(by='similarity_score', ascending=False).head(jumlah_rek)

    # 5. Visualisasi Hasil Rekomendasi Antarmuka
    st.info(f"Lagu ini masuk dalam kategori: **{nama_cluster}** (Cluster {target_cluster})")
    st.subheader("Daftar Rekomendasi Untukmu")

    html_cards = ""
    for index, row in rekomendasi_df.iterrows():
        match_percentage = round(row['similarity_score'] * 100, 1)
        
        html_cards += (
            '<div class="mini-song-card">'
                '<div>'
                    '<b style="line-height: 1.2;">' + str(row['track_name']) + '</b>'
                    '<span class="artist-text">👤 ' + str(row['artists']) + '</span>'
                '</div>'
                '<div>'
                    '<span class="stat-text" style="display: block; margin-bottom: 2px; color: #ffffff; font-weight: bold; background: rgba(0,0,0,0.15); padding: 2px 6px; border-radius: 6px; text-align: center;">🎯 Match: ' + str(match_percentage) + '%</span>'
                    '<span class="stat-text" style="display: block; margin-top: 4px; margin-bottom: 2px;">Popularity: ' + str(row['popularity']) + '</span>'
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

    # 6. Analisis Visual Dominasi Musisi
    st.subheader(f"Top 5 Artis di Kategori {nama_cluster}")
    st.write("Musisi yang paling mendominasi karakter suara di cluster ini.")

    top_artists = df[df['cluster'] == target_cluster]['artists'].value_counts().head(5).reset_index()
    top_artists.columns = ['Artist', 'Jumlah Lagu']
    top_artists = top_artists.iloc[::-1].reset_index(drop=True)

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

    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Data ditampilkan berdasarkan analisis Machine Learning terhadap dataset Spotify dari kaggle.com")

else:
    # Tampilan instruksi awal (placeholder sebelum tombol ditekan)
    st.info("**Selamat Datang!** Silakan cari lagu kesukaanmu di panel kiri, tentukan jumlah rekomendasi, lalu klik tombol **Cari Rekomendasi** untuk melihat hasilnya.")