# ============================================================
# app.py  —  STREAMLIT FRONTEND FOR BOOK RECOMMENDATION
# ============================================================

import streamlit as st
import pandas as pd

from backend import (
    MODEL_PATH,
    download_dataset,
    get_recommendations,
    load_data,
    load_saved_model,
    preprocess_data,
    train_and_save_model,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Book Recommendation System",
    layout="centered",
)

# ============================================================
# CUSTOM DARK THEME CSS
# ============================================================
# (Exact same theme as before — we're only changing WHAT the app
#  does, not HOW it looks.)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #050816, #0b1b34);
    color: white;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: #081120;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #0072ff, #00c6ff);
    color: white;
    border: none;
    border-radius: 10px;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0052cc, #0099cc);
    color: white;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    text-align: left;
    font-size: 18px;
    font-weight: bold;
    margin-top: 15px;
}

.success-box {
    background-color: rgba(46, 204, 113, 0.2);
    border: 2px solid #2ecc71;
}

.metric-card {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<h1 style='text-align:center;'>
Book Recommendation System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:#cccccc;'>
Pick a book you enjoyed, and we'll recommend similar ones for you to read next.
</p>
""", unsafe_allow_html=True)

# ============================================================
# LOAD SAVED RECOMMENDER (this is our "model")
# ============================================================
# @st.cache_resource means this heavy work only runs ONCE, even if
# the user clicks around the app a lot. That keeps the app fast.

@st.cache_resource
def get_recommender_bundle():
    if MODEL_PATH.exists():
        return load_saved_model()

    bundle, _ = train_and_save_model()
    return bundle


try:
    bundle = get_recommender_bundle()
except Exception as exc:
    st.error(
        "Recommender is not ready yet. Make sure your Kaggle API key is set up "
        "(see kagglehub docs) and run `python backend.py` once to build and save it."
    )
    st.exception(exc)
    st.stop()

books_df = bundle["books"]   # the cleaned table of all books


@st.cache_resource
def get_graph_data():
    csv_path = download_dataset()
    df = load_data(csv_path)
    df_clean = preprocess_data(df)
    return df, df_clean

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## Tell Us What You Like")

col1, col2 = st.columns(2)

with col1:

    # Let the user filter by genre first, to make picking a book easier
    all_genres = sorted(set(
        genre
        for genre_string in books_df["genres_clean"]
        for genre in genre_string.split(" ")
        if genre
    ))

    genre_filter = st.selectbox(
        "Filter by Genre (optional)",
        ["All Genres"] + all_genres
    )

with col2:

    top_n = st.slider(
        "How many recommendations?",
        min_value=3,
        max_value=10,
        value=5
    )

# Narrow the book list down if a genre was picked
if genre_filter == "All Genres":
    book_options_df = books_df
else:
    book_options_df = books_df[
        books_df["genres_clean"].str.contains(genre_filter, case=False, na=False)
    ]

book_choice = st.selectbox(
    "Pick a book you enjoyed",
    sorted(book_options_df["Book"].unique())
)

# ============================================================
# RECOMMEND BUTTON
# ============================================================

if st.button("Get Recommendations"):

    recommendations = get_recommendations(book_choice, bundle, top_n=top_n)

    st.markdown("## 📊 Recommended For You")

    if recommendations.empty:
        st.warning("Sorry, we couldn't find recommendations for that book.")
    else:
        for _, rec in recommendations.iterrows():
            st.markdown(f"""
            <div class="prediction-box success-box">
                 <a href="{rec['URL']}" target="_blank" style="color:#2ecc71;">{rec['Book']}</a><br>
                 {rec['Author']}<br>
                 {rec['Genres']}<br>
                 {rec['Avg_Rating']} / 5  ({rec['Num_Ratings']:,} ratings)<br>
                 Similarity to your book: {rec['Similarity %']}%
            </div>
            """, unsafe_allow_html=True)

        # ========================================================
        # METRICS FOR THE BOOK THE USER CHOSE
        # ========================================================

        st.markdown("## About Your Chosen Book")

        chosen_row = books_df[books_df["Book"] == book_choice].iloc[0]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Rating</h3>
                <h2>{chosen_row['Avg_Rating']}</h2>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3># Ratings</h3>
                <h2>{int(chosen_row['Num_Ratings']):,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Author</h3>
                <h2 style="font-size:16px;">{chosen_row['Author']}</h2>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Genres</h3>
                <h2 style="font-size:14px;">{chosen_row['genres_clean']}</h2>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#aaaaaa;'>

Built with using Streamlit

</div>
""", unsafe_allow_html=True)