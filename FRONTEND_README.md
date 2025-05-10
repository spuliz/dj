# News Article Semantic Search Frontend

This is a simple web interface for the news article semantic search system built with Streamlit.

## Features

- Search for news articles using natural language queries
- View articles by relevance with similarity scores
- Filter results by news category
- Color-coded results by category (World, Sports, Business, Science/Technology)
- Automatic caching of embeddings and indexes for faster performance

## Setup and Running

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run news_search_app.py
   ```

3. Your browser should automatically open to `http://localhost:8501`

## First-time Use

When you run the app for the first time, it will:
1. Download the AG News dataset
2. Create embeddings for ~5000 news articles
3. Build a FAISS index for fast search

This process may take a few minutes to complete. After the initial setup, subsequent launches will be much faster as the app loads the pre-calculated embeddings and index.

## Example Queries

Try searching for:
- "Latest developments in artificial intelligence"
- "Soccer match results and player transfers"
- "Stock market trends and economic outlook"
- "International politics and diplomatic relations"

## Technologies Used

- Streamlit for the web interface
- Sentence-BERT for creating semantic embeddings
- FAISS for efficient similarity search
- AG News dataset (120K news articles) 