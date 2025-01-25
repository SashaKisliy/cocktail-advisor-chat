import numpy as np


def extract_ingredients_from_query(query: str):
    if "recommend" in query.lower() and "containing" in query.lower():
        ingredients = query.lower().split("containing")[-1].strip().split(",")
        return [ingredient.strip() for ingredient in ingredients]
    return []


def recommend_cocktails(index, cocktails, ingredients):
    if not ingredients:
        return "Please specify ingredients for recommendations."

    favorites_vector = np.zeros(index.d, dtype='float32')
    for ingredient in ingredients:
        for cocktail in cocktails:
            if ingredient in cocktail['ingredients'].split(', '):
                favorites_vector += cocktail['vector']

    if np.linalg.norm(favorites_vector) != 0:
        favorites_vector /= np.linalg.norm(favorites_vector)

    distances, indices = index.search(np.array([favorites_vector]), k=5)
    recommended_cocktails = [cocktails[i]['name'] for i in indices[0]]
    return f"Recommended cocktails: {', '.join(recommended_cocktails)}"