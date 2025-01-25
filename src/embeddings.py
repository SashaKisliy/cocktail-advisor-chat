import numpy as np
import faiss


def create_faiss_index(ingredient_vectors: np.ndarray) -> faiss.IndexFlatL2:
    dimension = ingredient_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(ingredient_vectors)
    return index


def save_faiss_index(index: faiss.IndexFlatL2, file_path: str):
    faiss.write_index(index, file_path)


def load_faiss_index(file_path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(file_path)
