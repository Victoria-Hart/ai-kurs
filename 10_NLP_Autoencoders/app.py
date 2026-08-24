import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keras.models import load_model
from keras import Model


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "semantic_autoencoder_64d.keras"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "X_train_embeddings.npy"
LATENT_PATH = BASE_DIR / "data" / "processed" / "X_train_latent.npy"
DOCUMENTS_PATH = BASE_DIR / "data" / "processed" / "train_documents.pkl"


# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    autoencoder = load_model(
        MODEL_PATH
    )

    # Create the full encoder:
    # 384 → 256 → 128 → 64
    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("latent").output
    )

    return embedding_model, autoencoder, encoder


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    latent = np.load(
        LATENT_PATH
    )

    documents = pd.read_pickle(
        DOCUMENTS_PATH
    )

    return embeddings, latent, documents


embedding_model, autoencoder, encoder = load_models()

X_embeddings, X_latent, documents = load_data()


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.title("NLP Semantic Text Search")

st.write(
    "Enter a query to find semantically similar news articles "
    "using a compressed sentence representation."
)


query = st.text_input(
    "Search query",
    placeholder="e.g. Scientists develop new technology for space exploration"
)


top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5
)


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

if query:

    # Convert query into a 384-dimensional
    # sentence embedding
    query_embedding = embedding_model.encode(
        [query]
    )

    # Compress 384D → 64D using the trained encoder
    query_latent = encoder.predict(
        query_embedding,
        verbose=0
    )

    # Compare the query's 64D representation
    # with the stored 64D article representations
    similarities = cosine_similarity(
        query_latent,
        X_latent
    )[0]

    # Get indices of the most similar articles
    top_indices = similarities.argsort()[-top_k:][::-1]


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    st.subheader("Search Results")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        row = documents.iloc[index]

        st.markdown(
            f"### {rank}. {row['category']}"
        )

        st.write(
            row["text"]
        )

        st.caption(
            f"Similarity: {similarities[index]:.3f}"
        )

        st.divider()