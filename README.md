📚 Get Your Book — Recommender System

An interactive book recommender system built with Python, Streamlit, and Machine Learning (TF-IDF + KMeans).
It suggests similar books based on their titles, authors, genres, and descriptions, using Natural Language Processing (NLP) and unsupervised learning.

🚀 Features

✅ Book recommendations based on text similarity
✅ Fully interactive web app using Streamlit
✅ Filters by genre, author, and title search
✅ Pagination (24 books per page for faster performance)
✅ Displays book covers, titles, authors, and descriptions
✅ Automatic cover retrieval via ISBN using OpenLibrary API
✅ Full Machine Learning pipeline with preprocessing, vectorization, clustering, and recommendation

🧠 Machine Learning Pipeline
Step	Description
1️⃣ TF-IDF Vectorization	Transforms titles, authors, genres, and descriptions into numerical vectors
2️⃣ Normalization	Scales all vectors (L2 normalization)
3️⃣ KMeans Clustering	Groups books into 11 clusters based on textual similarity
4️⃣ Hybrid Recommendation	Combines cosine similarity (text) + metadata similarity (author/genre)
5️⃣ 2D Visualization (SVD)	Reduces dimensionality for cluster visualization

📊 Cluster Analysis

Optimal K = 11 clusters determined via Silhouette Score + Elbow Method

Each cluster represents a literary theme (e.g., thriller, romance, sci-fi)

SVD (Truncated Singular Value Decomposition) used to visualize clusters in 2D

Improved interpretability and insight into dataset structure

💻 Technologies Used
Category	Tools
Language	Python 3.11
Machine Learning	scikit-learn, NumPy, SciPy
NLP	TF-IDF, cosine similarity
Web App	Streamlit
Visualization	Matplotlib, Seaborn
Storage	joblib, CSV
Data Sources	Gutenberg, OpenLibrary API

📈 Results

3,347 books processed

20,000 textual features analyzed

11 thematic clusters identified


