import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =======================================================
# CONFIG
# =======================================================
st.set_page_config(page_title="📚 Get Your Book Recommender", layout="wide")

BASE_PATH = Path(__file__).resolve().parent
DATA_BASE = BASE_PATH / "data" / "raw" / "processed"
DATA_PATH = DATA_BASE / "ml_outputs"


# =======================================================
# LOAD DATA
# =======================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_BASE / "books_final_version.csv")
    df["title"] = df["title"].astype(str).str.strip()
    df["author"] = df["author"].fillna("Unknown").astype(str)
    df["genre"] = df["genre"].fillna("other").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["url"] = df["url"].fillna("")
    df["cover_image_uri"] = df["cover_image_uri"].fillna(
        "https://via.placeholder.com/150x220?text=No+Cover"
    )

    df["uid"] = df["isbn13"].astype(str).where(df["isbn13"].notna(), df.index.astype(str))
    return df.reset_index(drop=True)


@st.cache_resource
def load_models():
    tfidf = joblib.load(DATA_PATH / "tfidf_vectorizer.joblib")
    X_tfidf = load_npz(DATA_PATH / "tfidf_matrix_norm.npz")
    return tfidf, X_tfidf


df = load_data()
tfidf, X_tfidf = load_models()


# =======================================================
# RECOMMENDER (HÍBRIDO)
# =======================================================
def recommend_books(title, top_n=10, alpha=0.75):
    idx_match = df.index[df["title"].str.lower() == title.lower()]
    if len(idx_match) == 0:
        return pd.DataFrame()

    idx = idx_match[0]

    sims_text = cosine_similarity(X_tfidf[idx], X_tfidf).flatten()

    sims_meta = np.zeros_like(sims_text, dtype=float)
    sims_meta += (df["author"].str.lower() == df.loc[idx, "author"].lower()).astype(float)
    sims_meta += (df["genre"].str.lower() == df.loc[idx, "genre"].lower()).astype(float)

    sims = alpha * sims_text + (1 - alpha) * sims_meta
    sims[idx] = -1

    top_idx = sims.argsort()[::-1][:top_n]
    out = df.loc[top_idx].copy()
    out["score"] = sims[top_idx]
    return out


# =======================================================
# STATE
# =======================================================
ss = st.session_state
ss.setdefault("page", "home")
ss.setdefault("selected_uid", None)


# =======================================================
# RENDER CARD
# =======================================================
def render_card(row):
    st.image(row.cover_image_uri, width=150)

    st.markdown(f"### {row.title}", unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center; margin-top:-10px; font-weight:600'>{row.author}</div>",
        unsafe_allow_html=True,
    )

    st.caption(row.genre)

    short_desc = row.description[:180] + "…" if len(row.description) > 180 else row.description
    st.write(short_desc)

    colA, colB = st.columns(2)

    with colA:
        if st.button("📘 View details", key=f"details_{row.uid}"):
            ss.page = "details"
            ss.selected_uid = row.uid

    with colB:
        if st.button("🔮 More like this", key=f"more_{row.uid}"):
            ss.page = "details"
            ss.selected_uid = row.uid


# =======================================================
# PAGE: HOME
# =======================================================
def page_home():
    st.title("📚 Get Your Book Recommender")

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        genres = ["All"] + sorted(df["genre"].unique().tolist())
        selected_genre = st.selectbox("🎭 Filter by genre", genres)

    with col2:
        authors = ["All"] + sorted(df["author"].unique().tolist())
        selected_author = st.selectbox("👤 Filter by author", authors)

    with col3:
        search = st.text_input("🔎 Search by title", "")

    filtered = df.copy()
    if selected_genre != "All":
        filtered = filtered[filtered["genre"] == selected_genre]
    if selected_author != "All":
        filtered = filtered[filtered["author"] == selected_author]
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False)]

    st.markdown(f"### {len(filtered)} books found")

    books = list(filtered.itertuples(index=False))

    for i in range(0, len(books), 4):
        cols = st.columns(4)
        for col, row in zip(cols, books[i:i+4]):
            with col:
                render_card(row)


# =======================================================
# PAGE: DETAILS
# =======================================================
def page_details():
    uid = ss.selected_uid
    book = df[df["uid"] == uid].iloc[0]

    st.button("⬅️ Back to home", on_click=lambda: ss.update({"page": "home"}))

    st.header(book.title)
    st.caption(f"👤 {book.author}")
    st.image(book.cover_image_uri, width=200)
    st.write(book.description)

    st.subheader("🔮 Recommended books")
    recs = recommend_books(book.title, top_n=10)

    for i in range(0, len(recs), 4):
        cols = st.columns(4)
        for col, row in zip(cols, recs.iloc[i:i+4].itertuples(index=False)):
            with col:
                render_card(row)


# =======================================================
# ROUTER
# =======================================================
if ss.page == "home":
    page_home()
elif ss.page == "details":
    page_details()
else:
    page_home()
