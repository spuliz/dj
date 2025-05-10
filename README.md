# Semantic Search Project

This project implements a semantic search solution using the Stanford Question Answering Dataset (SQuAD) to create an effective question-answering retrieval system.

## Project Overview

The system uses transformer-based embeddings to encode questions and passages, then performs semantic similarity search to retrieve the most relevant passages for a given question.

### Key Components:
- Sentence-BERT for creating semantic embeddings
- FAISS for efficient similarity search
- Evaluation using Recall@k and Mean Reciprocal Rank

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```
   jupyter notebook
   ```
4. Open `notebooks/semantic_search_project.ipynb` to explore the implementation

## Dataset

The project uses the Stanford Question Answering Dataset (SQuAD), a reading comprehension dataset consisting of questions posed on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.

## Project Structure

```
├── data/               # Data directory (dataset will be downloaded via HuggingFace)
├── notebooks/          # Jupyter notebooks for implementation
│   └── semantic_search_project.ipynb
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Evaluation Metrics

The system is evaluated using:
- Recall@k: Measures whether the relevant passage is among the top k retrieved results
- Mean Reciprocal Rank (MRR): Measures the ranking quality of the retrieval system