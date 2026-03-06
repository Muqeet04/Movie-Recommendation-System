import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Cache the data loading so it doesn't reload on every click
@st.cache_data
def load_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    df = pd.merge(ratings, movies, on='movieId')
    user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    return user_movie_matrix

# Cache the SVD computation
@st.cache_resource
def compute_svd(user_movie_matrix):
    X = user_movie_matrix.T
    SVD = TruncatedSVD(n_components=12, random_state=42)
    matrix_decomposed = SVD.fit_transform(X)
    corr_matrix = np.corrcoef(matrix_decomposed)
    return corr_matrix, list(user_movie_matrix.columns)

# UI Setup
st.title("🎬 Movie Recommendation Engine")
st.markdown("Enter a movie you like, and we'll find similar titles using SVD Matrix Factorization.")

try:
    user_movie_matrix = load_data()
    corr_matrix, movie_names = compute_svd(user_movie_matrix)

    # Sidebar/Search
    selected_movie = st.selectbox("Type or select a movie:", movie_names)

    if st.button('Recommend'):
        idx = movie_names.index(selected_movie)
        correlation_movie_ID = corr_matrix[idx]
        
        # Get indices of top correlations (excluding the movie itself)
        # We look for correlations > 0.9 or just take the top 10
        similar_indices = np.where(correlation_movie_ID > 0.9)[0]
        recommendations = [movie_names[i] for i in similar_indices if movie_names[i] != selected_movie]

        if not recommendations:
            st.warning("No highly similar movies found. Try another title!")
        else:
            st.subheader(f"If you liked '{selected_movie}', you might also like:")
            cols = st.columns(2)
            for i, movie in enumerate(recommendations[:10]):
                cols[i % 2].write(f"✅ {movie}")

except FileNotFoundError:
    st.error("Please make sure 'ratings.csv' and 'movies.csv' are in the same directory.")
