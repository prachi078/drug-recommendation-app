import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import os
import gdown
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')

# Google Drive file IDs
CSV_FILE_ID = '1fg2Gt1RetHFk8wi6eEdhlOme-MR7AgaP'
NPY_FILE_ID = '1vRwABVNQmDYcII0XrTpJLwmLI0xSa26N'

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# Load data and model
@st.cache_resource
def load_resources():
    # Download files if they don't exist
    if not os.path.exists('processed_data.csv'):
        download_file_from_google_drive(CSV_FILE_ID, 'processed_data.csv')
    if not os.path.exists('sentence_embeddings.npy'):
        download_file_from_google_drive(NPY_FILE_ID, 'sentence_embeddings.npy')
    
    df = pd.read_csv('processed_data.csv')
    embeddings = np.load('sentence_embeddings.npy')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, embeddings, model

df, embeddings, model = load_resources()

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", '', str(text))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

# Recommendation function
def recommend_drugs(user_input, top_n=5, min_similarity=0.0):
    cleaned_input = preprocess_text(user_input)
    user_embedding = model.encode([cleaned_input])
    cosine_sim = cosine_similarity(user_embedding, embeddings).flatten()

    results = pd.DataFrame({
        'drugName': df['drugName'],
        'similarity_score': cosine_sim
    }).sort_values(by='similarity_score', ascending=False)

    # Filter by similarity threshold
    results = results[results['similarity_score'] >= min_similarity]

    return results.head(top_n)

# Streamlit UI
st.title("💊 Drug Recommendation System (SBERT)")
st.write("Enter your symptoms or condition description.")

# Inputs
user_input = st.text_area("Describe your condition or symptoms:")
top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
min_similarity = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

if st.button("Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing and fetching recommendations..."):
            results = recommend_drugs(user_input, top_n, min_similarity)

        st.success("Top Recommendations:")
        st.dataframe(results.reset_index(drop=True))

        # CSV Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='drug_recommendations.csv',
            mime='text/csv',
        )
