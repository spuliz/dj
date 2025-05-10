import streamlit as st
import numpy as np
import pandas as pd
import torch
import requests
import io
import csv
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import time

# Set page title and layout
st.set_page_config(
    page_title="News Semantic Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🔍 News Article Semantic Search")
st.markdown("""
This application demonstrates semantic search for news articles using transformer-based embeddings.
Enter your query to find the most relevant news articles from the AG News dataset.
""")

# Define category mapping
category_mapping = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}

# Function to download AG News dataset directly
def download_ag_news(max_samples=5000):
    # Create data directory if needed
    os.makedirs('data', exist_ok=True)
    
    # Check if we already have the CSV files
    train_file = 'data/ag_news_train.csv'
    
    if os.path.exists(train_file):
        st.sidebar.success("✅ Using existing AG News dataset.")
    else:
        # Direct URLs to AG News dataset
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        
        st.sidebar.info("⏳ Downloading AG News dataset...")
        
        # Download training data
        response = requests.get(train_url)
        with open(train_file, 'wb') as f:
            f.write(response.content)
        
        st.sidebar.success("✅ AG News dataset downloaded successfully!")
    
    # Load and process the dataset
    df = pd.read_csv(train_file, header=None)
    df.columns = ['class_id', 'title', 'content']
    
    # Convert 1-based to 0-based indexing for class IDs
    df['class_id'] = df['class_id'] - 1
    
    # Add category names
    df['category'] = df['class_id'].map(category_mapping)
    
    # Combine title and content
    df['article'] = df['title'] + ". " + df['content']
    
    # Take only max_samples
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
    
    return df

# Function to load model and data
@st.cache_resource
def load_resources():
    try:
        # Load model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Try to load existing embeddings and index
        embeddings_file = 'data/train_article_embeddings.pkl'
        index_file = 'data/news_faiss_index.bin'
        
        if os.path.exists(embeddings_file) and os.path.exists(index_file):
            # Load embeddings
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Load index
            index = faiss.read_index(index_file)
            
            # Load dataset
            df = download_ag_news()
            
            st.sidebar.success("✅ Loaded existing embeddings and index.")
            return model, index, df
        
        # If embeddings or index doesn't exist, create them
        st.sidebar.warning("⚠️ Building index from scratch (this may take a few minutes)...")
        
        # Load dataset
        df = download_ag_news(max_samples=5000)
        
        # Create embeddings
        embeddings = model.encode(df['article'].tolist(), batch_size=16, show_progress_bar=True)
        
        # Save embeddings
        os.makedirs('data', exist_ok=True)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Create index
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        normalized_embeddings = embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        index.add(normalized_embeddings)
        faiss.write_index(index, index_file)
        
        st.sidebar.success("✅ Successfully created new embeddings and index.")
        return model, index, df
        
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.stop()

# Function to perform semantic search
def semantic_news_search(query, model, index, df, top_k=5):
    # Encode the query
    query_embedding = model.encode([query])
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    scores, indices = index.search(query_embedding, top_k)
    
    # Return results
    results = []
    for i, idx in enumerate(indices[0]):
        idx = int(idx)  # Convert from numpy.int64 to avoid issues
        results.append({
            'article': df.iloc[idx]['article'],
            'category': df.iloc[idx]['category'],
            'score': float(scores[0][i]),
            'index': idx
        })
    
    return results

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
### News Semantic Search

This demo uses:
- AG News dataset (120K news articles)
- Sentence-BERT for embeddings
- FAISS for similarity search
- Streamlit for the interface

You can search across articles from 4 categories:
- World
- Sports
- Business
- Science/Technology
""")

# Load model, index and data
with st.spinner("Loading resources..."):
    model, index, df = load_resources()

# Query input
query = st.text_input("Enter your search query:", placeholder="e.g., latest developments in artificial intelligence")

# Number of results selector
num_results = st.slider("Number of results to display:", min_value=1, max_value=10, value=5)

# Search button
if st.button("Search") or query:
    if query:
        with st.spinner(f"Searching for: '{query}'"):
            start_time = time.time()
            results = semantic_news_search(query, model, index, df, top_k=num_results)
            search_time = time.time() - start_time
        
        # Display results
        st.success(f"Found {len(results)} results in {search_time:.3f} seconds")
        
        # Display category filters
        st.subheader("Filter by Category")
        all_categories = sorted(list(category_mapping.values()))
        selected_categories = st.multiselect("Select categories:", all_categories, default=all_categories)
        
        # Filter results by selected categories
        if selected_categories:
            filtered_results = [r for r in results if r['category'] in selected_categories]
        else:
            filtered_results = []
        
        st.subheader("Search Results")
        
        # Display results with color-coded categories
        for i, result in enumerate(filtered_results):
            col1, col2 = st.columns([1, 5])
            
            # Determine category color
            category = result['category']
            if category == "World":
                category_color = "blue"
            elif category == "Sports":
                category_color = "green"
            elif category == "Business":
                category_color = "orange"
            else:  # Science/Technology
                category_color = "purple"
            
            # Display result with category badge and score
            with col1:
                st.markdown(f"<span style='background-color:{category_color};color:white;padding:4px 8px;border-radius:4px;'>{category}</span>", unsafe_allow_html=True)
                st.markdown(f"**Score: {result['score']:.3f}**")
            
            with col2:
                # Extract title (first sentence) and content
                title = result['article'].split('.')[0]
                content = '.'.join(result['article'].split('.')[1:])
                
                # Display title and truncated content
                st.markdown(f"**{title}.**")
                if len(content) > 300:
                    st.markdown(f"{content[:300]}...")
                else:
                    st.markdown(content)
            
            # Add separator between results
            st.markdown("---")
    else:
        st.info("Please enter a search query.")

# Footer
st.markdown("---")
st.markdown("Semantic Search for News Media | AG News Dataset") 