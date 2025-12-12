
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------
# Global variables
# ------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
segment_list = []

# Segment management
# ----------------------
def set_segments(segments):
    global segment_list
    segment_list = segments

def get_segment_by_index(idx):
    return segment_list[idx]["text"] if idx < len(segment_list) else ""

# Embeddings
# ------------------
def get_embeddings(segments):
    texts = [seg["text"] for seg in segments]
    embs = embedding_model.encode(texts, convert_to_numpy=True)
    return embs
# ------------
# FAISS Index
# -------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index, path="data/faiss_index.bin"):
    faiss.write_index(index, path)

def load_faiss_index(path="data/faiss_index.bin"):
    return faiss.read_index(path)

# ------------------
# Semantic Search
# ---------------------
def search_similar(query, top_k=3):
    query_vec = embedding_model.encode([query])
    index = load_faiss_index()
    distances, indices = index.search(query_vec, top_k)
    return distances, indices

# -------------
# Testcode
# -----------------
if __name__ == "__main__":
    sample_segments = [
        {"start": 0.0, "end": 5.0, "text": "I worked on machine learning models."},
        {"start": 5.0, "end": 10.0, "text": "I improved system performance."},
        {"start": 10.0, "end": 15.0, "text": "I deployed models to production."}
    ]

    set_segments(sample_segments)
    embs = get_embeddings(sample_segments)
    print("Embeddings shape:", embs.shape)

    index = build_faiss_index(embs)
    print("FAISS index built:", index.ntotal)

    save_faiss_index(index)
    print("FAISS index saved.")

    # Test search
    distances, idxs = search_similar("machine learning")
    print("Distances:", distances)
    print("Indices:", idxs)
