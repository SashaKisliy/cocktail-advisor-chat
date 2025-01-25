from src.preprocessing import load_cocktails, preprocess_ingredients
from src.embeddings import create_faiss_index, save_faiss_index


def prepare_faiss_index(csv_path: str, index_path: str):
    cocktails = load_cocktails(csv_path)
    ingredient_vectors, _ = preprocess_ingredients(cocktails)

    index = create_faiss_index(ingredient_vectors)
    save_faiss_index(index, index_path)
