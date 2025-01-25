import pandas as pd
import numpy as np


def load_cocktails(file_path: str) -> pd.DataFrame:
    """
    Загружает данные о коктейлях из CSV.
    """
    df = pd.read_csv(file_path)
    df['ingredients'] = df['ingredients'].fillna('')
    return df


def preprocess_ingredients(cocktails: pd.DataFrame) -> (np.ndarray, dict):
    all_ingredients = set()
    for ingredients in cocktails['ingredients']:
        all_ingredients.update(ingredients.split(', '))

    ingredient_to_index = {ingredient: idx for idx, ingredient in enumerate(sorted(all_ingredients))}

    ingredient_vectors = []
    for ingredients in cocktails['ingredients']:
        vector = np.zeros(len(ingredient_to_index), dtype='float32')
        for ingredient in ingredients.split(', '):
            if ingredient in ingredient_to_index:
                vector[ingredient_to_index[ingredient]] = 1.0
        ingredient_vectors.append(vector)

    return np.array(ingredient_vectors), ingredient_to_index
