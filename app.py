import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Import untuk grafik interaktif
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- AI Libraries ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ==========================================
# 1. KONFIGURASI & CSS (UI/UX UPGRADE)
# ==========================================
st.set_page_config(
    page_title="CineMatch AI - Hybrid Engine",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan Modern/Premium
st.markdown("""
<style>
    /* Styling Container Utama */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Hero Section untuk Film Terpilih */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Card Rekomendasi */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e1e1e;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    
    /* AI Insight Box */
    .ai-insight-box {
        background-color: #13151A;
        border-left: 4px solid #FF4B4B;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin-top: 20px;
    }
    
    /* Judul Section */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        margin-top: 2rem;
        color: #f8fafc;
        border-bottom: 2px solid #FF4B4B;
        display: inline-block;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SETUP API KEYS ---
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ API Keys belum disetting! Harap buat file .streamlit/secrets.toml")
    st.stop()

# ==========================================
# 3. BACKEND LOGIC (TIDAK DIUBAH)
# ==========================================

# --- A. CRAWLING ---
@st.cache_data(show_spinner=False)
def fetch_tmdb_data(api_key, pages):
    base_url = "https://api.themoviedb.org/3"
    genre_map = {}
    
    # Custom Progress Bar di Sidebar agar tidak mengganggu UI utama
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        g_res = requests.get(f"{base_url}/genre/movie/list?api_key={api_key}&language=en-US")
        if g_res.status_code == 200:
            genre_map = {g['id']: g['name'] for g in g_res.json()['genres']}
    except: pass

    movies = []
    
    for page in range(1, pages + 1):
        status_text.text(f"⏳ Downloading Page {page}/{pages}...")
        progress_bar.progress(page / pages)
        
        url = f"{base_url}/movie/popular?api_key={api_key}&language=en-US&page={page}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                for m in res.json().get('results', []):
                    g_names = [genre_map.get(gid, "") for gid in m.get('genre_ids', [])]
                    movies.append({
                        'id': m.get('id'),
                        'title': m.get('title'),
                        'overview': m.get('overview', ''),
                        'genres': ", ".join(g_names),
                        'popularity': m.get('popularity', 0.0),
                        'vote_average': m.get('vote_average', 0.0),
                        'release_date': m.get('release_date', 'Unknown'),
                        'poster_path': f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else None
                    })
            time.sleep(0.05)
        except: pass
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(movies).drop_duplicates(subset=['id']).reset_index(drop=True)
    return df[df['overview'].str.len() > 10]

# --- B. HYBRID MODELING ---
@st.cache_resource
def build_hybrid_model(df):
    df['content_features'] = df['title'] + " " + df['genres'] + " " + df['overview']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    
    scaler = MinMaxScaler()
    df[['vote_scaled', 'pop_scaled']] = scaler.fit_transform(df[['vote_average', 'popularity']])

    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    numeric_sim = cosine_similarity(df[['vote_scaled', 'pop_scaled']], df[['vote_scaled', 'pop_scaled']])

    # Hybrid Logic (70% Content, 30% Quality)
    hybrid_sim = (0.7 * content_sim) + (0.3 * numeric_sim)
    
    return df, hybrid_sim

# --- C. SEARCH LOGIC ---
def get_search_and_recommend_logic(movie_query, df, hybrid_sim, top_n=5):
    query_norm = movie_query.lower().strip()
    df['title_lower'] = df['title'].str.lower().str.strip()

    search_results = df[df['title_lower'].str.contains(query_norm, na=False)].copy()
    msg = f"Hasil pencarian: '{movie_query}'"

    if search_results.empty:
        all_titles = df['title_lower'].tolist()
        matches = difflib.get_close_matches(query_norm, all_titles, n=1, cutoff=0.5)
        
        if matches:
            query_norm = matches[0]
            search_results = df[df['title_lower'].str.contains(query_norm, na=False)].copy()
            msg = f"Typo terdeteksi. Mungkin maksud Anda: '{query_norm}'?"
        else:
            return None, None, f"Maaf, film '{movie_query}' tidak ditemukan di database."

    anchor_idx = None
    exact_match = df[df['title_lower'] == query_norm]

    if not exact_match.empty:
        anchor_idx = exact_match.index[0]
    elif not search_results.empty:
        anchor_idx = search_results.index[0]

    rec_df = None
    if anchor_idx is not None:
        sim_scores = list(enumerate(hybrid_sim[anchor_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        rec_indices = [i[0] for i in sim_scores[1:top_n+1]] 

        rec_df = df.iloc[rec_indices][['title', 'genres', 'vote_average', 'release_date', 'overview', 'poster_path']].copy()
        rec_df['similarity_score'] = [i[1] for i in sim_scores[1:top_n+1]]
    
    display_search = search_results[['title', 'genres', 'vote_average', 'release_date', 'overview', 'poster_path', 'popularity']]
    return display_search, rec_df, msg

# --- D. AI EVALUATION ---
class MovieEval(BaseModel):
    score: int = Field(description="Skor 1-10")
    analysis: str = Field(description="Analisis singkat")
    verdict: str = Field(description="Kesimpulan pendek")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0)
eval_parser = PydanticOutputParser(pydantic_object=MovieEval)

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah Kritikus Film AI. Nilai rekomendasi berikut berdasarkan Genre, Plot, dan Mood."),
    ("human", """
    FILM DISUKAI USER: "{input_movie}"
    REKOMENDASI SISTEM: {recommendations}

    Berikan skor 1-10 dan alasan kritis.
    {format_instructions}
    """)
])

eval_chain = eval_prompt | llm | eval_parser

def evaluate_with_ai(anchor_title, rec_df):
    try:
        rec_list = rec_df[['title', 'genres']].to_dict(orient='records')
        return eval_chain.invoke({
            "input_movie": anchor_title,
            "recommendations": rec_list,
            "format_instructions": eval_parser.get_format_instructions()
        })
    except Exception as e:
        return f"⚠️ Gagal evaluasi AI: {str(e)}"

# ==========================================
# 4. FRONTEND UI (LAYOUT UTAMA)
# ==========================================

# --- Sidebar (Sudah Bagus - Dipertahankan) ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    num_pages = st.slider("Jumlah Data (Pages):", 10, 100, 30, 10, help="Semakin banyak page, semakin lengkap rekomendasi.")
    
    st.info("💡 **Tips:** Jika rekomendasi kurang relevan, coba tambah jumlah page dan refresh data.")
    
    if st.button("🔄 Refresh / Reload Data", type="secondary"):
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("---")
    st.caption("🔍 Hybrid Filtering:\n- 70% Content (TF-IDF)\n- 30% Quality (Rating/Pop)")

# --- Main Content ---
st.title("🍿 CineMatch AI")
st.markdown("##### Temukan film favoritmu berikutnya dengan kekuatan Hybrid AI & Gemini.")

# Load Data
df_raw = fetch_tmdb_data(TMDB_API_KEY, num_pages)
df, hybrid_sim = build_hybrid_model(df_raw)

# --- Search Area (Clean & Centered) ---
st.markdown("<br>", unsafe_allow_html=True)
col_search, col_btn = st.columns([6, 1], gap="small")

with col_search:
    query = st.text_input("Search", placeholder="Ketik judul film (Contoh: Inception, Avengers, Frozen)...", label_visibility="collapsed")

with col_btn:
    search_clicked = st.button("🔍 Cari", type="primary", use_container_width=True)

# --- Logic Execution ---
if query or search_clicked:
    if not query:
        st.toast("⚠️ Mohon ketik judul film terlebih dahulu.", icon="⚠️")
    else:
        # Panggil Logic
        search_df, rec_df, msg = get_search_and_recommend_logic(query, df, hybrid_sim)

        if search_df is None:
            st.error(msg)
        else:
            # === HERO SECTION (HASIL PENCARIAN) ===
            target = search_df.iloc[0]
            
            st.markdown(f'<div class="section-title">📂 {msg}</div>', unsafe_allow_html=True)
            
            # Layout Hero 2 Kolom
            col_hero_img, col_hero_txt = st.columns([1, 3])
            
            with col_hero_img:
                if target['poster_path']:
                    st.image(target['poster_path'], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
            
            with col_hero_txt:
                st.markdown(f"""
                <div class="hero-container">
                    <h1 style='margin:0; font-size: 2.5em;'>{target['title']}</h1>
                    <p style='color: #cbd5e1; font-style: italic;'>Rilis: {target['release_date']}</p>
                    <hr style='border-color: #475569;'>
                    <p style='font-size: 1.1em; line-height: 1.6;'>{target['overview']}</p>
                    <br>
                    <div style='display: flex; gap: 20px;'>
                        <div>⭐ <b>Rating:</b> {target['vote_average']}/10</div>
                        <div>📈 <b>Popularitas:</b> {int(target['popularity'])}</div>
                        <div>🎭 <b>Genre:</b> {target['genres']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # === REKOMENDASI SECTION (GRID CARD) ===
            if rec_df is not None:
                st.markdown('<div class="section-title">✨ Rekomendasi Hybrid Pilihan</div>', unsafe_allow_html=True)
                
                cols = st.columns(5)
                for i, (idx, row) in enumerate(rec_df.iterrows()):
                    with cols[i]:
                        # Card Style Container
                        with st.container(border=True):
                            if row['poster_path']:
                                st.image(row['poster_path'], use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                            
                            # Judul (Truncate jika terlalu panjang)
                            short_title = (row['title'][:25] + '..') if len(row['title']) > 25 else row['title']
                            st.markdown(f"**{short_title}**")
                            
                            # Score Metric
                            st.metric("Hybrid Score", f"{row['similarity_score']:.2f}")
                            
                            with st.popover("Sinopsis"):
                                st.write(row['overview'])

                # === AI INSIGHT SECTION ===
                st.markdown('<div class="section-title">🤖 Analisis Cerdas Gemini AI</div>', unsafe_allow_html=True)
                
                if st.button("✨ Minta Pendapat AI tentang Rekomendasi ini"):
                    with st.spinner("Gemini sedang menganalisis selera filmmu..."):
                        eval_res = evaluate_with_ai(target['title'], rec_df)
                    
                    if isinstance(eval_res, MovieEval):
                        # Dynamic Color based on Score
                        score_color = "#4CAF50" if eval_res.score >= 8 else ("#FFC107" if eval_res.score >= 6 else "#FF4B4B")
                        
                        st.markdown(f"""
                        <div class="ai-insight-box">
                            <h2 style='margin:0; color: {score_color};'>Skor Kecocokan: {eval_res.score}/10</h2>
                            <h4 style='margin-top:5px; margin-bottom: 20px;'><i>"{eval_res.verdict}"</i></h4>
                            <p style='font-size: 1.1em;'>{eval_res.analysis}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(eval_res)

# ==========================================
# 5. FOOTER: EDA DENGAN SWITCH
# ==========================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📊 Statistik & Eksplorasi Data (EDA)"):
    # Header EDA
    st.markdown("#### Distribusi Data Film")
    
    # SWITCH: STATIC VS DYNAMIC
    viz_mode = st.radio(
        "Mode Visualisasi:", 
        ["Interaktif (Plotly)", "Statik (Seaborn)"], 
        horizontal=True,
        help="Pilih 'Interaktif' untuk grafik yang bisa di-zoom dan hover."
    )
    
    col1, col2 = st.columns(2)
    
    # 1. VISUALISASI RATING
    with col1:
        st.caption("Distribusi Rating Penonton")
        if viz_mode == "Interaktif (Plotly)":
            fig1 = px.histogram(df, x="vote_average", nbins=20, 
                                title="Histogram Rating (Interaktif)",
                                color_discrete_sequence=['#636EFA'])
            fig1.update_layout(bargap=0.1)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['vote_average'], bins=20, kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)

    # 2. VISUALISASI POPULARITAS VS RATING
    with col2:
        st.caption("Hubungan Popularitas vs Rating")
        if viz_mode == "Interaktif (Plotly)":
            fig2 = px.scatter(df, x="vote_average", y="popularity", 
                              hover_data=['title'], 
                              title="Scatter Plot Popularitas (Interaktif)",
                              color_discrete_sequence=['#EF553B'])
            fig2.update_yaxes(range=[0, 500]) # Batasi outlier agar rapi
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='vote_average', y='popularity', alpha=0.6, color='orange', ax=ax)
            ax.set_ylim(0, 500)
            st.pyplot(fig)
            
    # 3. WORDCLOUD (Selalu Statik karena library WordCloud berbasis gambar)
    st.markdown("#### WordCloud Sinopsis")
    text = " ".join(review for review in df.overview.astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)
    
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)