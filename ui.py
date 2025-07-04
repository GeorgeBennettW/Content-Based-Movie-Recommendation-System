import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import load_npz

# --- Load all saved models and data ---
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    knn_model = joblib.load('knn_model.pkl')
    tfidf_matrix = load_npz('tfidf_matrix.npz')
    movies = pd.read_csv('dataset/movies_cleaned.csv')
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return tfidf, knn_model, tfidf_matrix, movies, indices

tfidf, knn_model, tfidf_matrix, movies, indices = load_models()

# --- Recommender function ---
def recommend_knn(title, top_n=10):
    idx = indices.get(title)
    if idx is None:
        return None, f"Movie '{title}' not found in the dataset."
    
    movie_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(movie_vector, n_neighbors=top_n + 1)
    
    similar_indices = indices_knn[0][1:]
    similar_movies = movies.iloc[similar_indices][['title', 'genres']].copy()
    similar_movies['similarity'] = 1 - distances[0][1:]
    return similar_movies.reset_index(drop=True), None

# --- Streamlit UI ---
st.title("Content-Based Movie Recommender")
st.write("Find movies similar to your favorite title based on genres and tags.")

movie_titles = sorted(movies['title'].unique())
selected_movie = st.selectbox("Select a movie:", movie_titles)
top_n = st.slider("How many recommendations?", min_value=1, max_value=20, value=10)

if st.button("Recommend"):
    recommendations, error = recommend_knn(selected_movie, top_n=top_n)
    if error:
        st.error(error)
    else:
        st.success(f"Top {top_n} movies similar to '{selected_movie}':")
        st.dataframe(recommendations)