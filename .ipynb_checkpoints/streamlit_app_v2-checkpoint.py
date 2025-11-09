import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="📚 Get Your Book Recommender", layout="wide")

# ✅ caminho correto
DATA_PATH = Path("/home/rafael/Área de Trabalho/CUROS IRONHACK/SEMANA 10/get_your_book/data/raw/processed/ml_outputs")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH / "books_clustered.csv")
    df["title"] = df["title"].astype(str).str.strip()
    df["author"] = df["author"].astype(str).fillna("Unknown")
    df["description"] = df["description"].fillna("")
    df["genre"] = df.get("genre", "other").fillna("other")
    df["cluster_label"] = df.get("cluster_label", "unknown").astype(str)
    df["image_url"] = df.get("image_url", None)
    df["uid"] = df.get("isbn13", df.index).astype(str)
    return df.sort_values("title").reset_index(drop=True)

@st.cache_resource
def load_models():
    tfidf = joblib.load(DATA_PATH / "tfidf_vectorizer.joblib")
    X_tfidf = load_npz(DATA_PATH / "tfidf_matrix_norm.npz")
    return tfidf, X_tfidf

df = load_data()
tfidf, X_tfidf = load_models()

# =====================================
# RECOMMENDER
# =====================================
def recommend_books(base_title, top_n=5):
    base_idx = df[df["title"].str.lower() == base_title.lower()]
    if base_idx.empty:
        return pd.DataFrame()
    idx = base_idx.index[0]
    sims = cosine_similarity(X_tfidf[idx], X_tfidf).flatten()
    top_idx = sims.argsort()[::-1][1:top_n+1]
    return df.loc[top_idx, ["title", "author", "genre", "avg_rating", "cluster_label", "url", "description", "image_url"]]

# =====================================
# ESTADO
# =====================================
if "page" not in st.session_state:
    st.session_state.page = 1
if "selected_book" not in st.session_state:
    st.session_state.selected_book = None
if "page_changed" not in st.session_state:
    st.session_state.page_changed = False

def go_prev():
    st.session_state.page = max(1, st.session_state.page - 1)
    st.session_state.selected_book = None
    st.session_state.page_changed = True

def go_next(total_pages):
    st.session_state.page = min(total_pages, st.session_state.page + 1)
    st.session_state.selected_book = None
    st.session_state.page_changed = True

def reset_all():
    st.session_state.page = 1
    st.session_state.selected_book = None
    st.session_state.page_changed = True

# =====================================
# UI TABS
# =====================================
tab1, tab2 = st.tabs(["📖 Explore Books", "📊 Cluster Insights"])

# =====================================
# TAB 1 — CATÁLOGO + RECOMENDAÇÕES
# =====================================
with tab1:
    st.title("📚 Get Your Book Recommender")
    st.caption("Explore livros e descubra recomendações personalizadas com base em título, autor e descrição.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        genres = sorted(df["genre"].dropna().unique())
        selected_genre = st.selectbox("🎭 Filter by genre:", ["All"] + genres, on_change=reset_all)
    with col_f2:
        search = st.text_input("🔎 Search books by title", placeholder="Type part of a title...", on_change=reset_all)

    # filtros
    books_filtered = df.copy()
    if selected_genre != "All":
        books_filtered = books_filtered[books_filtered["genre"] == selected_genre]
    if search:
        books_filtered = books_filtered[books_filtered["title"].str.contains(search, case=False, na=False)]

    # paginação
    books_per_page = 20
    total_pages = max(1, (len(books_filtered) + books_per_page - 1) // books_per_page)
    page = st.session_state.page
    start = (page - 1) * books_per_page
    end = start + books_per_page
    books_page = books_filtered.iloc[start:end]

    st.markdown(f"### {len(books_filtered)} books found — showing page {page} of {total_pages}")

    if len(books_page) == 0:
        st.warning("No books found with these filters.")
    else:
        for _, row in books_page.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(row["image_url"] if pd.notna(row.get("image_url")) else "https://via.placeholder.com/100x150?text=No+Cover", width=100)
                with cols[1]:
                    st.markdown(f"### {row['title'].title()}")
                    st.caption(f"👤 {row['author']}")
                    if len(row["description"]) > 0:
                        st.write(row["description"][:280] + "...")
                    if pd.notna(row.get("url")):
                        st.markdown(f"[🔗 Access book]({row['url']})")

                    if not st.session_state.page_changed:
                        if st.button("🔮 See recommendations", key=f"rec_{row['uid']}"):
                            st.session_state.selected_book = row["title"]
                    else:
                        st.button("🔮 See recommendations", key=f"rec_{row['uid']}", disabled=True)

    # controles de página
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.button("⬅️ Previous", disabled=(page == 1), on_click=go_prev)
    with col2:
        st.markdown(f"<div style='text-align:center'>Page <b>{page}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
    with col3:
        st.button("Next ➡️", disabled=(page == total_pages), on_click=lambda: go_next(total_pages))

    # recomendações (só se a página não acabou de mudar)
    if st.session_state.selected_book and not st.session_state.page_changed:
        title = st.session_state.selected_book
        st.subheader(f"🔮 Recommendations for: *{title}*")
        recs = recommend_books(title, top_n=5)
        if recs.empty:
            st.warning("No recommendations found.")
        else:
            for _, rec in recs.iterrows():
                with st.container(border=True):
                    cols = st.columns([1, 4])
                    with cols[0]:
                        st.image(rec["image_url"] if pd.notna(rec.get("image_url")) else "https://via.placeholder.com/100x150?text=No+Cover", width=100)
                    with cols[1]:
                        st.markdown(f"**{rec['title'].title()}** — *{rec['author']}*")
                        st.caption(f"🎭 {rec['genre']} | ⭐ {rec['avg_rating'] if pd.notna(rec['avg_rating']) else 'N/A'} | 🧩 {rec['cluster_label']}")
                        st.write(rec["description"][:320] + "...")
                        if pd.notna(rec.get("url")):
                            st.markdown(f"[🔗 Access book]({rec['url']})")

    st.session_state.page_changed = False

# =====================================
# TAB 2 — INSIGHTS (Apenas KMeans + Silhouette)
# =====================================
with tab2:
    st.header("📊 Cluster Analysis — TF-IDF + K-Means")

    X = X_tfidf
    Ks = list(range(4, 16))
    silhouette_scores = []

    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouette_scores.append(sil)

    # gráfico silhouette
    st.subheader("📈 Silhouette Score by K")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=Ks, y=silhouette_scores, marker="o", ax=ax)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Method for Optimal K")
    st.pyplot(fig)

    best_k = Ks[int(np.argmax(silhouette_scores))]
    st.success(f"Best K = {best_k} (Silhouette = {max(silhouette_scores):.4f})")

    # Top terms
    st.subheader("💡 Top terms per cluster")
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(X)
    feature_names = np.array(tfidf.get_feature_names_out())
    centroids = kmeans_final.cluster_centers_

    top_terms = []
    for i in range(best_k):
        top_idx = centroids[i].argsort()[-10:][::-1]
        top_terms.append({"cluster": i, "top_terms": ", ".join(feature_names[top_idx])})
    st.dataframe(pd.DataFrame(top_terms))

    # Contagem de livros por cluster
    st.subheader("📚 Books per cluster")
    df["cluster_label_recalc"] = labels_final.astype(str)
    st.bar_chart(df["cluster_label_recalc"].value_counts().sort_index())
