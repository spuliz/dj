# News Semantic Search Project

This project implements a semantic search solution for news articles to enhance content discovery and recommendation in media publishing.

## Project Overview

The system uses transformer-based embeddings to encode news articles and user queries, then performs semantic similarity search to retrieve the most relevant news content. This enables more effective content discovery beyond simple keyword matching.

### Key Features:
- Semantic search for news articles using natural language queries
- Category-based organization and filtering (World, Sports, Business, Science/Technology)
- Color-coded search results with relevance scoring
- Cross-category content discovery
- Interactive web interface built with Streamlit

## Live Demo

You can access a live demo of this semantic search system at:
[https://9flabixjy2bbz7fvhfyzgg.streamlit.app/](https://9flabixjy2bbz7fvhfyzgg.streamlit.app/)

Try it out to explore the capabilities of the system without needing to run the code locally!

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run news_search_app.py
   ```
4. Your browser should automatically open to `http://localhost:8501`

## Dataset

The project uses the AG News Corpus, a collection of news articles across four categories:
- World
- Sports
- Business
- Science/Technology

The dataset is automatically downloaded the first time you run the application.

## Implementation Details

### Pipeline:
1. **Data Preprocessing**: Load AG News articles, extract text and categories
2. **Feature Engineering**: Create dense vector embeddings for news content using Sentence-BERT
3. **Indexing**: Build a FAISS index for efficient similarity search
4. **Search**: Encode user queries and retrieve the most semantically similar articles
5. **Evaluation**: Measure relevance and category precision

### Technologies Used:
- Sentence-Transformers for creating semantic embeddings
- FAISS for efficient similarity search
- Streamlit for the interactive web interface
- Pandas & NumPy for data manipulation

## Example Queries

Try searching for:
- "Latest developments in artificial intelligence"
- "Soccer match results and player transfers"
- "Stock market trends and economic outlook"
- "International politics and diplomatic relations"

## Practical Applications

This semantic search system can be integrated into news platforms to:
1. Enhance search functionality for readers
2. Power "related articles" recommendations
3. Create topic-based content collections
4. Support journalists in research by finding relevant past coverage
5. Increase user engagement through better content discovery