# NLP Semantic Text Search

A semantic text search application using sentence embeddings and an autoencoder to compress the embedding representation while preserving retrieval performance.

## Live Demo

[Try the Streamlit app](https://semantic-search-news.streamlit.app/)

## Project Overview

This project explores whether sentence embeddings can be compressed into a lower-dimensional representation without substantially reducing semantic retrieval performance.

The project uses the **AG News** dataset and the `all-MiniLM-L6-v2` sentence-transformer model to generate 384-dimensional sentence embeddings.

A neural-network autoencoder then compresses these embeddings from 384 dimensions to a 64-dimensional latent representation.

The compressed representations are used for semantic search with cosine similarity.

## Approach

The project consists of the following steps:

1. Load and inspect the AG News dataset.
2. Select a balanced training subset of 20,000 articles.
3. Generate 384-dimensional sentence embeddings using `all-MiniLM-L6-v2`.
4. Train an autoencoder to compress the embeddings to 64 dimensions.
5. Visualize the latent space using PCA and UMAP.
6. Compare semantic retrieval using the original and compressed representations.
7. Deploy the compressed semantic search system as a Streamlit application.

## Results

The original and compressed representations were evaluated using Precision@5 and Precision@10.

| Representation | Dimensions | Precision@5 | Precision@10 |
|---|---:|---:|---:|
| Sentence embeddings | 384 | 0.857 | 0.845 |
| Autoencoder latent space | 64 | 0.849 | 0.836 |

The autoencoder reduces the representation from **384 to 64 dimensions**, an approximately **83.3% reduction in dimensionality**.

Despite this reduction, the compressed representation retains approximately:

- **99.1%** of the original Precision@5 performance
- **98.9%** of the original Precision@10 performance

This suggests that the autoencoder preserves most of the information relevant to topical semantic retrieval while substantially reducing the dimensionality of the representation.

## Technologies

- Python
- TensorFlow / Keras
- Sentence Transformers
- NumPy
- Pandas
- Scikit-learn
- UMAP
- Matplotlib / Seaborn
- Streamlit

## Project Structure

```text
NLP-Autoencoders-Semantic-Search/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── processed/
│       ├── X_train_embeddings.npy
│       ├── X_test_embeddings.npy
│       ├── X_train_latent.npy
│       ├── X_test_latent.npy
│       └── train_documents.pkl
│
├── models/
│   ├── semantic_autoencoder_64d.keras
│   └── semantic_autoencoder_64d_history.json
│
└── notebooks/
    └── NLP_Autoencoders_Semantic_Text_Search.ipynb
