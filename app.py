import os, zipfile, urllib.request
import numpy as np, pandas as pd, streamlit as st
from sklearn.decomposition import TruncatedSVD

st.title("CineMatch: Movie Recommender")
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

@st.cache_data
def load():
    if not os.path.exists("ml-latest-small"):
        urllib.request.urlretrieve(URL, "ml.zip")
        zipfile.ZipFile("ml.zip").extractall(".")
    return (pd.read_csv("ml-latest-small/ratings.csv"),
            pd.read_csv("ml-latest-small/movies.csv"))

@st.cache_resource
def build():
    ratings, movies = load()
    users = np.sort(ratings.userId.unique())
    items = np.sort(ratings.movieId.unique())
    u = ratings.userId.map({x: i for i, x in enumerate(users)}).values
    i = ratings.movieId.map({x: j for j, x in enumerate(items)}).values
    R = np.zeros((len(users), len(items)), dtype=np.float32)
    R[u, i] = ratings.rating.values
    M = R > 0
    user_mean = R.sum(1) / np.maximum(M.sum(1), 1)
    Rc = (R - user_mean[:, None]) * M
    svd = TruncatedSVD(n_components=50, random_state=42).fit(Rc)
    pred = user_mean[:, None] + svd.inverse_transform(svd.transform(Rc))
    return users, items, R, M, pred, movies.set_index("movieId")

users, items, R, M, pred, meta = build()

user_id = st.selectbox("Choose a user ID", users)
n = st.slider("How many recommendations?", 3, 15, 5)
u = list(users).index(user_id)

st.subheader("Movies this user rated highest")
liked = np.argsort(-R[u])[:5]
for j in liked:
    st.write("- " + meta.loc[items[j], "title"])

scores = pred[u].copy()
scores[M[u]] = -np.inf
scores[M.sum(0) < 20] = -np.inf
best = np.argsort(-scores)[:n]

st.subheader("We recommend")
for j in best:
    st.write(f"- {meta.loc[items[j], 'title']} (predicted {np.clip(scores[j], 0.5, 5):.2f})")