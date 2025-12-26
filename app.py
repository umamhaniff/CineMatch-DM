import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import difflib
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn still used for static visuals
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Streamlit page configuration
st.set_page_config(
    page_title="CineMatch",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
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
            color: var(--text-color);
            border-bottom: 2px solid #FF4B4B;
            display: inline-block;
            padding-bottom: 5px;
        }

        .h4{
            color: var(--text-color);
        }

        /* Fixed-size movie cards for search results and recommendations */
        .movie-card,
        .recommend-card {
            /* Set a fixed height to ensure all cards align, leaving room for synopsis */
            min-height: 500px;
            height: auto;
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            border: 1px solid #334155;
            padding: 0.5rem;
            /* margin: top right bottom left: minimal horizontal, larger bottom for row separation */
            margin: 0.5rem 1rem 0.5rem 1rem;
            box-sizing: border-box;
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .movie-card img,
        .recommend-card img {
            width: 100%;
            height: 280px;
            object-fit: cover;
            border-radius: 8px;
        }
        .movie-card h4,
        .recommend-card h4 {
            margin: 0.5rem 0 0.25rem;
            font-size: 1rem;
            flex-shrink: 0;
        }
        .movie-card small,
        .recommend-card small {
            display: block;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        .recommend-card .score {
            margin-top: 0.5rem;
            font-weight: bold;
            color: #fbbf24;
        }
        /* Synopsis styling inside cards (unused but kept for potential future use) */
        .movie-card .synopsis,
        .recommend-card .synopsis {
            margin-top: auto;
            font-size: 0.9rem;
            line-height: 1.4;
            color: #d1d5db;
            overflow-y: auto;
            max-height: 100px;
        }
        .movie-card details.synopsis,
        .recommend-card details.synopsis {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background-color: #111827;
            border-radius: 8px;
            color: #94a3b8;
        }
        .movie-card details.synopsis summary,
        .recommend-card details.synopsis summary {
            cursor: pointer;
            font-weight: bold;
            color: #f1f5f9;
            outline: none;
        }
        /* Flex container for hero section to ensure equal height columns */
        .hero-flex {
            display: flex;
            gap: 20px;
        }
        .hero-flex .hero-img {
            flex: 1;
        }
        .hero-flex .hero-details {
            flex: 3;
        }
        .hero-flex .hero-img img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Keys configuration
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ API Keys belum disetting! Harap buat file .streamlit/secrets.toml")
    st.stop()

# Data acquisition from TMDB
def fetch_tmdb_data(api_key: str, pages: int) -> pd.DataFrame:
    """
    Crawl popular movies from TMDB API while displaying a progress bar and status.

    Parameters
    ----------
    api_key : str
        TMDB API key.
    pages : int
        Number of pages to crawl (each page contains up to 20 movies).

    Returns
    -------
    pd.DataFrame
        DataFrame containing movie metadata with non-empty overviews.
    """
    base_url = "https://api.themoviedb.org/3"
    genre_map: dict[int, str] = {}
    # Progress bar and status text will appear wherever this function is called
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Fetch genre mapping
    try:
        g_res = requests.get(f"{base_url}/genre/movie/list?api_key={api_key}&language=en-US")
        if g_res.status_code == 200:
            genre_map = {g['id']: g['name'] for g in g_res.json()['genres']}
    except Exception:
        pass

    movies: list[dict] = []

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
            # polite delay between requests
            time.sleep(0.05)
        except Exception:
            pass
    # Clear progress bar and status
    progress_bar.empty()
    status_text.empty()

    df = pd.DataFrame(movies).drop_duplicates(subset=['id']).reset_index(drop=True)
    return df[df['overview'].str.len() > 10]

# Hybrid content and numeric similarity model
@st.cache_resource
def build_hybrid_model(df: pd.DataFrame):
    """
    Build a hybrid similarity model based on TF-IDF content features and numeric quality features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing movie metadata. Must contain columns 'title', 'genres', 'overview',
        'vote_average', and 'popularity'.

    Returns
    -------
    tuple
        The processed dataframe (with new columns) and the hybrid similarity matrix.
    """
    df['content_features'] = df['title'] + " " + df['genres'] + " " + df['overview']
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['content_features'])

    scaler = MinMaxScaler()
    df[['vote_scaled', 'pop_scaled']] = scaler.fit_transform(df[['vote_average', 'popularity']])

    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    numeric_sim = cosine_similarity(df[['vote_scaled', 'pop_scaled']], df[['vote_scaled', 'pop_scaled']])

    hybrid_sim = (0.7 * content_sim) + (0.3 * numeric_sim)
    return df, hybrid_sim

# Search logic and recommendation retrieval
def get_search_and_recommend_logic(movie_query: str, df: pd.DataFrame, hybrid_sim: np.ndarray, top_n: int = 5):
    """
    Perform a fuzzy search on movie titles and compute recommendations based on the hybrid similarity matrix.

    Parameters
    ----------
    movie_query : str
        User input query for movie title.
    df : pd.DataFrame
        DataFrame containing movie metadata and processed features.
    hybrid_sim : np.ndarray
        Precomputed hybrid similarity matrix.
    top_n : int, optional
        Number of recommendations to return, by default 5.

    Returns
    -------
    tuple
        (search_results, rec_df, msg)
    """
    query_norm = movie_query.lower().strip()
    df['title_lower'] = df['title'].str.lower().str.strip()

    search_results = df[df['title_lower'].str.contains(query_norm, na=False)].copy()
    msg = f"Hasil pencarian: '{movie_query}'"

    # handle typos
    if search_results.empty:
        all_titles = df['title_lower'].tolist()
        matches = difflib.get_close_matches(query_norm, all_titles, n=1, cutoff=0.5)
        if matches:
            query_norm = matches[0]
            search_results = df[df['title_lower'].str.contains(query_norm, na=False)].copy()
            msg = f"Typo terdeteksi. Mungkin maksud Anda: '{query_norm}'?"
        else:
            return None, None, f"Maaf, film '{movie_query}' tidak ditemukan di database."

    # select anchor: exact match prioritized
    anchor_idx = None
    exact_match = df[df['title_lower'] == query_norm]

    if not exact_match.empty:
        anchor_row = exact_match.index[0]
        anchor_idx = df.index.get_loc(anchor_row)
    elif not search_results.empty:
        anchor_row = search_results.index[0]
        anchor_idx = df.index.get_loc(anchor_row)

    if anchor_idx is None or anchor_idx >= hybrid_sim.shape[0]:
        return search_results, None, "Film ditemukan, tetapi tidak cukup data untuk rekomendasi."

    rec_df = None
    if anchor_idx is not None:
        sim_scores = list(enumerate(hybrid_sim[anchor_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        rec_indices = [i[0] for i in sim_scores[1:min(top_n+1, len(sim_scores))]]
        rec_df = df.iloc[rec_indices][['title', 'genres', 'vote_average', 'release_date', 'overview', 'poster_path']].copy()
        rec_df['similarity_score'] = [sim_scores[i+1][1] for i in range(len(rec_indices))]

    if anchor_idx is None or hybrid_sim is None or len(df) <= 1:
        return search_results, None, "Film ditemukan, tetapi tidak cukup data untuk rekomendasi."

    display_search = search_results[['title', 'genres', 'vote_average', 'release_date', 'overview', 'poster_path', 'popularity']]
    return display_search, rec_df, msg

# AI evaluation class and chain
class MovieEval(BaseModel):
    score: int = Field(description="Skor 1-10")
    analysis: str = Field(description="Analisis singkat")
    verdict: str = Field(description="Kesimpulan pendek")

# Daftar model fallback untuk mengatasi rate limit
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-flash"]

# Parser dan prompt tetap sama
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

def evaluate_with_ai(anchor_title: str, rec_df: pd.DataFrame):
    """
    Fungsi ini mencoba beberapa model Gemini secara berurutan.
    Jika satu model terkena rate limit, akan mencoba model berikutnya.
    """
    rec_list = rec_df[['title', 'genres']].to_dict(orient='records')
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0
            )
            eval_chain = eval_prompt | llm | eval_parser
            return eval_chain.invoke({
                "input_movie": anchor_title,
                "recommendations": rec_list,
                "format_instructions": eval_parser.get_format_instructions()
            })
        except Exception as e:
            last_error = e
            continue
    return f"⚠️ Tidak dapat memproses permintaan: {last_error}"

# Sidebar information panel
with st.sidebar:
    st.header("ℹ️ Ringkasan Data")
    # Display data summary if available
    if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None and not st.session_state['df_raw'].empty:
        df_side = st.session_state['df_raw']
        total_movies = len(df_side)
        avg_rating = df_side['vote_average'].mean() if not df_side['vote_average'].empty else 0
        # extract release years ignoring unknowns
        years = pd.to_datetime(df_side['release_date'], errors='coerce').dt.year.dropna().astype(int)
        if not years.empty:
            year_min, year_max = years.min(), years.max()
        else:
            year_min, year_max = None, None
        # compute top genres
        genre_counts_side = df_side['genres'].str.split(', ').explode().value_counts()
        top_genres = ", ".join(genre_counts_side.head(3).index) if not genre_counts_side.empty else "-"
        st.metric("Jumlah Film", total_movies)
        st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
        if year_min is not None:
            st.metric("Periode Rilis", f"{year_min} - {year_max}")
        else:
            st.write("Periode Rilis: Tidak diketahui")
        st.write(f"**Top Genre:** {top_genres}")
    else:
        st.write("Belum ada data yang diambil. Silakan lakukan crawling data.")

    st.markdown("---")
    st.caption("🔍 Hybrid Filtering:\n- 70% Content (TF-IDF)\n- 30% Quality (Rating/Pop)")


# Main application logic
st.title("🍿 CineMatch")
st.markdown("##### Temukan film favoritmu berikutnya dengan kekuatan Hybrid AI & Gemini.")

# Initialize session state for data and evaluation
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'hybrid_sim' not in st.session_state:
    st.session_state['hybrid_sim'] = None
if 'pages' not in st.session_state:
    st.session_state['pages'] = 50  # default pages
if 'force_reload' not in st.session_state:
    st.session_state['force_reload'] = True  # initial load
if 'evaluation' not in st.session_state:
    st.session_state['evaluation'] = None
if 'data_loading' not in st.session_state:
    st.session_state['data_loading'] = False
if 'top_n' not in st.session_state:
    # default number of recommendations
    st.session_state['top_n'] = 5


# Placeholder for progress bar beneath title and search area
progress_placeholder = st.empty()

# Automatically load default data on first run if no data is present and not already loading
if st.session_state.get('df_raw') is None and not st.session_state['data_loading']:
    st.session_state['data_loading'] = True
    with progress_placeholder:
        df_init = fetch_tmdb_data(TMDB_API_KEY, st.session_state['pages'])
        if df_init is None or df_init.empty:
            st.error("❌ Tidak ada film yang berhasil dimuat dari TMDB.")
            st.session_state['data_loading'] = False
            st.stop()
    df_proc_init, hybrid_init = build_hybrid_model(df_init)
    st.session_state['df_raw'] = df_init
    st.session_state['df'] = df_proc_init
    st.session_state['hybrid_sim'] = hybrid_init
    st.session_state['data_loading'] = False
    progress_placeholder.empty()

st.markdown("<br>", unsafe_allow_html=True)

# Search bar row with recommendation count and settings popover
col_search, col_topn, col_btn, col_settings = st.columns([6, 1, 1, 1], gap="small")
with col_search:
    query = st.text_input(
        "Search",
        placeholder="Ketik judul film (Contoh: Inception, Avengers, Frozen)...",
        label_visibility="collapsed",
    )
with col_topn:
    # Control for number of recommendations; step of 5
    topn_input = st.number_input(
        label="Jumlah Rekomendasi",
        min_value=5,
        max_value=50,
        value=int(st.session_state.get('top_n', 5)),
        step=5,
        format="%d",
        help="Jumlah film rekomendasi yang ingin ditampilkan.",
        label_visibility="collapsed"
    )
    st.session_state['top_n'] = int(topn_input)
with col_btn:
    # Disable search when data is loading or not yet loaded
    disabled_search = (st.session_state.get('df') is None or st.session_state['data_loading'])
    search_clicked = st.button(
        "🔍 Cari",
        type="primary",
        use_container_width=True,
        disabled=disabled_search
    )
with col_settings:
    # Settings popover for crawling configuration; gear icon only
    with st.popover("⚙️"):
        st.markdown("Atur jumlah halaman film populer yang ingin diambil dari TMDB.")
        pages_slider = st.slider(
            "Jumlah Data (Pages):",
            10,
            100,
            st.session_state['pages'],
            10,
            help="Semakin banyak page, semakin lengkap rekomendasi."
        )
        if st.button("🔄 Muat / Refresh Data", key="refresh_data_btn"):
            st.session_state['data_loading'] = True
            st.session_state['pages'] = pages_slider
            st.session_state['evaluation'] = None
            # Display progress bar during data fetching
            with progress_placeholder:
                df_new = fetch_tmdb_data(TMDB_API_KEY, pages_slider)
                if df_new is None or df_new.empty:
                    st.error("❌ Data kosong, refresh dibatalkan.")
                    st.stop()
            df_processed, hybrid_sim_new = build_hybrid_model(df_new)
            st.session_state['df_raw'] = df_new
            st.session_state['df'] = df_processed
            st.session_state['hybrid_sim'] = hybrid_sim_new
            st.session_state['data_loading'] = False
            progress_placeholder.empty()
            st.success("✅ Data berhasil dimuat!", icon="✅")

# Retrieve current data from session state for local use
df_raw = st.session_state.get('df_raw')
df = st.session_state.get('df')
hybrid_sim = st.session_state.get('hybrid_sim')

if query or search_clicked:
    # Ensure data is loaded before performing any search
    if df is None or hybrid_sim is None:
        st.info("Data belum dimuat. Silakan muat data terlebih dahulu melalui ikon pengaturan.")
    else:
        # Validate that query is not empty after stripping whitespace
        if not query.strip():
            st.toast("⚠️ Mohon ketik judul film terlebih dahulu.", icon="⚠️")
        else:
            search_df, rec_df, msg = get_search_and_recommend_logic(
                query,
                df,
                hybrid_sim,
                top_n=int(st.session_state.get('top_n', 5))
            )
            if search_df is None:
                st.warning(msg)
                st.info("💡 Coba judul lain atau gunakan kata kunci yang lebih umum.")
                st.stop()
            else:
                # Reset evaluation whenever a new search is performed
                st.session_state['evaluation'] = None
                # Display the search header message
                st.markdown(f'<div class="section-title">📂 {msg}</div>', unsafe_allow_html=True)
                # Hero section highlighting the first search result with equal height columns
                target = search_df.iloc[0]
                # Determine poster URL or fallback
                hero_poster = target['poster_path'] if target['poster_path'] else "https://via.placeholder.com/300x450?text=No+Image"
                hero_html = f"""
                    <div class="hero-flex">
                        <div class="hero-img">
                            <img src="{hero_poster}" alt="Poster">
                        </div>
                        <div class="hero-details">
                            <div class="hero-container">
                                <h1 style='margin:0; font-size: 2.5em;'>{target['title']}</h1>
                                <p style='color: #cbd5e1; font-style: italic;'>Rilis: {target['release_date']}</p>
                                <hr style='border-color: #475569;'>
                                <p style='font-size: 1.1em; line-height: 1.6;'>{target['overview']}</p>
                                <br>
                                <div style='display: flex; gap: 20px;'>
                                    <div>⭐ <b>Rating:</b> {target['vote_average']}/10</div>
                                    <div>🔥 <b>Popularitas:</b> {int(target['popularity'])}</div>
                                    <div>🎭 <b>Genre:</b> {target['genres']}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                """
                st.markdown(hero_html, unsafe_allow_html=True)
                # Card-based display of all search results
                if len(search_df) > 0:
                    st.markdown('<div class="section-title">📋 Hasil Pencarian</div>', unsafe_allow_html=True)
                    results_df = search_df.reset_index(drop=True)
                    # Always allocate 5 columns; leftover items will align to the left
                    max_cols = 5
                    for start in range(0, len(results_df), max_cols):
                        sub_df = results_df.iloc[start:start + max_cols]
                        cols = st.columns(max_cols, gap=None)
                        for j, row_data in enumerate(sub_df.itertuples(index=False)):
                            with cols[j]:
                                poster_url = row_data.poster_path if getattr(row_data, 'poster_path') else "https://via.placeholder.com/300x450?text=No+Image"
                                title_trunc = (row_data.title[:25] + '..') if len(row_data.title) > 25 else row_data.title
                                # Construct card without embedded synopsis; popularity uses fire emoji
                                card_html = f"""
                                    <div class="movie-card">
                                        <img src="{poster_url}" alt="Poster">
                                        <h4>{title_trunc}</h4>
                                        <small>⭐ {row_data.vote_average}/10 | 🔥 {int(row_data.popularity)}</small>
                                        <small>🎭 {row_data.genres}</small>
                                        <small>📅 {row_data.release_date}</small>
                                        <details class="synopsis">
                                            <summary>ℹ️ Sinopsis</summary>
                                            <p>{row_data.overview}</p>
                                        </details>
                                    </div>
                                """
                                st.markdown(card_html, unsafe_allow_html=True)
                # Display recommendations if available
                if rec_df is not None and not rec_df.empty:
                    st.markdown('<div class="section-title">✨ Rekomendasi Hybrid Pilihan</div>', unsafe_allow_html=True)
                    rec_count = len(rec_df)
                    max_cols = 5
                    # Generate rows of recommendations, left-aligning leftover items
                    for start in range(0, rec_count, max_cols):
                        sub_df = rec_df.iloc[start:start + max_cols]
                        cols = st.columns(max_cols, gap=None)
                        for j, row in enumerate(sub_df.itertuples(index=False)):
                            with cols[j]:
                                poster_url = row.poster_path if getattr(row, 'poster_path') else "https://via.placeholder.com/300x450?text=No+Image"
                                short_title = (row.title[:25] + '..') if len(row.title) > 25 else row.title
                                card_html = f"""
                                    <div class="recommend-card">
                                        <img src="{poster_url}" alt="Poster">
                                        <h4>{short_title}</h4>
                                        <small>🎭 {row.genres}</small>
                                        <small>📅 {row.release_date}</small>
                                        <small class="score">Hybrid Score: {row.similarity_score:.2f}</small>
                                        <details class="synopsis">
                                            <summary>ℹ️ Sinopsis</summary>
                                            <p>{row.overview}</p>
                                        </details>
                                    </div>
                                """
                                st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.info("😔 Belum ada rekomendasi serupa untuk film ini.")
                    st.caption("Dataset mungkin terlalu kecil atau film ini unik.")
                # AI evaluation section
                st.markdown('<div class="section-title">🤖 Analisis Cerdas Gemini AI</div>', unsafe_allow_html=True)
                if rec_df is not None:
                    # Show button if evaluation hasn't been done yet
                    if st.session_state['evaluation'] is None:
                        if rec_df is None or rec_df.empty:
                            st.info("🤖 AI tidak dijalankan karena tidak ada film rekomendasi.")
                        else:
                            if st.button("✨ Minta Pendapat AI tentang Rekomendasi ini"):
                                with st.spinner("Gemini sedang menganalisis selera filmmu..."):
                                    eval_res = evaluate_with_ai(target['title'], rec_df)
                                st.session_state['evaluation'] = eval_res
                    # Display evaluation results if available
                    if st.session_state['evaluation'] is not None:
                        eval_res = st.session_state['evaluation']
                        if isinstance(eval_res, MovieEval):
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=eval_res.score,
                                gauge={'axis': {'range': [0, 10]}},
                                title={'text': 'AI Score'}
                            ))
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            st.markdown(
                                f"### Skor Kecocokan: {eval_res.score}/10 ({eval_res.verdict})"
                            )
                            st.markdown(
                                f"{eval_res.analysis}"
                            )
                        else:
                            st.error(eval_res)
                else:
                    st.info("Tidak ada rekomendasi untuk film ini.")
else:
    st.info("Masukkan judul film untuk memulai pencarian.")

# Exploratory Data Analysis (EDA) section using expander
st.markdown("---")

with st.expander("📊 Statistik & Eksplorasi Data (EDA)"):
    if df is None or df.empty:
        st.info("Belum ada data untuk dieksplorasi. Silakan muat data terlebih dahulu.")
    else:
        st.markdown("#### Distribusi Data Film")
        viz_mode = st.radio(
            "Mode Visualisasi:",
            ["Interaktif (Plotly)", "Statik (Seaborn)"],
            horizontal=True,
            help="Pilih 'Interaktif' untuk grafik yang bisa di-zoom dan hover."
        )
        # Three-column layout: histogram, scatter, and top genre
        col1, col2, col3 = st.columns(3)
        # Rating distribution (Histogram)
        with col1:
            st.caption("Distribusi Rating Penonton")
            if viz_mode == "Interaktif (Plotly)":
                fig1 = px.histogram(
                    df,
                    x="vote_average",
                    nbins=20,
                    title="Histogram Rating"
                )
                fig1.update_layout(bargap=0.1)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                fig_h, ax_h = plt.subplots(figsize=(6, 4))
                sns.histplot(df['vote_average'], bins=20, kde=True, ax=ax_h)
                ax_h.set_title("Histogram Rating")
                st.pyplot(fig_h)
        # Popularity vs Rating (Scatter)
        with col2:
            st.caption("Hubungan Popularitas vs Rating")
            if viz_mode == "Interaktif (Plotly)":
                fig2 = px.scatter(
                    df,
                    x="vote_average",
                    y="popularity",
                    hover_data=['title'],
                    title="Popularitas vs Rating"
                )
                fig2.update_yaxes(range=[0, 500])
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig_s, ax_s = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df, x='vote_average', y='popularity', alpha=0.6, ax=ax_s)
                ax_s.set_ylim(0, 500)
                ax_s.set_title("Popularitas vs Rating")
                st.pyplot(fig_s)
        # Top 10 genres bar chart
        with col3:
            st.caption("Top 10 Genre")
            genre_counts = df['genres'].str.split(', ').explode().value_counts().head(10)
            if viz_mode == "Interaktif (Plotly)":
                fig_genres = px.bar(
                    x=genre_counts.index,
                    y=genre_counts.values,
                    labels={'x': 'Genre', 'y': 'Jumlah'},
                    title="Top 10 Genre"
                )
                st.plotly_chart(fig_genres, use_container_width=True)
            else:
                fig_gen, ax_gen = plt.subplots(figsize=(6, 4))
                ax_gen.bar(genre_counts.index, genre_counts.values)
                ax_gen.set_title("Top 10 Genre")
                ax_gen.set_xticklabels(genre_counts.index, rotation=45, ha='right')
                st.pyplot(fig_gen)
        # WordCloud displayed centered below
        st.markdown("#### WordCloud Sinopsis")
        text = " ".join(review for review in df.overview.astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)
        wc_img = wordcloud.to_array()
        # Center the wordcloud using column layout
        wc_col1, wc_col2, wc_col3 = st.columns([1, 3, 1])
        with wc_col2:
            st.image(wc_img, width=600)