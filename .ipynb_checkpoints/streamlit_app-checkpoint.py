import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="📚 Get Your Book", layout="centered")

DATA_PATH = Path("/home/rafael/Área de Trabalho/CUROS IRONHACK/SEMANA 10/get_your_book/data/raw/processed/ml_outputs")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH / "books_clustered.csv")
    df["title"] = df["title"].astype(str).str.strip().str.lower()
    df["author"] = df["author"].astype(str).str.strip()
    df["description"] = df["description"].fillna("")
    return df

@st.cache_resource
def load_models():
    tfidf = joblib.load(DATA_PATH / "tfidf_vectorizer.joblib")
    X_tfidf = load_npz(DATA_PATH / "tfidf_matrix_norm.npz")
    return tfidf, X_tfidf

df = load_data()
tfidf, X_tfidf = load_models()


# Recommendation function

def recommend_books(title_query, top_n=5):
    title_query = title_query.lower().strip()
    matches = df[df["title"].str.contains(title_query, case=False, na=False)]
    if matches.empty:
        st.warning(f"Any book found: '{title_query}' 😢")
        return None

    idx = matches.index[0]
    book_vec = X_tfidf[idx]
    sims = cosine_similarity(book_vec, X_tfidf).flatten()
    top_idx = sims.argsort()[::-1][1:top_n+1]

    recs = df.loc[top_idx, ["title", "author", "genre", "avg_rating", "cluster_label", "url", "description"]]
    st.markdown(f"### 📖 Base book: *{df.loc[idx, 'title'].title()}* — {df.loc[idx, 'author']}")
    st.divider()

    for _, row in recs.iterrows():
        with st.container():
            st.markdown(f"**{row['title'].title()}** — *{row['author']}*")
            st.markdown(f"📚 Genre: `{row['genre']}` | ⭐ {row['avg_rating'] if pd.notna(row['avg_rating']) else 'N/A'}")
            st.markdown(f"🧩 Cluster: `{row['cluster_label']}`")
            if pd.notna(row['url']):
                st.markdown(f"[🔗 Access link]({row['url']})")
            if pd.notna(row['description']) and len(row['description']) > 0:
                st.caption(row['description'][:300] + "...")
            st.markdown("---")


# Main Interface

st.title("📚 Get Your Book Recommender")
st.markdown("Find similar books based on *title, author and description!*")

title_query = st.text_input("Type a book name:", placeholder="Example: Frankenstein, Pride and Prejudice, Moby Dick...")
top_n = st.slider("How many recommendations do you like to see?", 3, 10, 5)

if st.button("🔍 Search recommendations"):
    recommend_books(title_query, top_n)

st.markdown("---")
st.caption("Developed by Rafael Cabral — Ironhack 2025")
