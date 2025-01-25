from src.database import prepare_faiss_index, load_cocktails
from src.embeddings import load_faiss_index
from src.rag import recommend_cocktails, extract_ingredients_from_query
from src.llm import ask_openai

DATA_PATH = "../data/final_cocktails.csv"
INDEX_PATH = "../data/cocktail_index.faiss"

def load_or_prepare_index():
    try:
        index = load_faiss_index(INDEX_PATH)
        print("FAISS index loaded successfully.")
    except Exception:
        print("FAISS index not found. Preparing a new one...")
        prepare_faiss_index(DATA_PATH, INDEX_PATH)
        index = load_faiss_index(INDEX_PATH)
    return index

def handle_question(question, index, cocktails):
    # Extract user preferences if needed
    ingredients = extract_ingredients_from_query(question)

    if "recommend" in question.lower():
        if ingredients:
            return recommend_cocktails(index, cocktails, ingredients)
        elif "similar to" in question.lower():
            return "Here's a cocktail similar to 'Hot Creamy Bush': [Your Cocktail Name Here]"
        else:
            return "Please specify what you need recommendations for."
    else:
        return ask_openai(question)

def main():
    # Load or prepare FAISS index
    index = load_or_prepare_index()

    # Load cocktails data
    cocktails = load_cocktails(DATA_PATH).to_dict(orient='records')

    print("Welcome to the Cocktail Advisor Chat CLI!")
    while True:
        question = input("Ask a question (or 'exit' to quit): ")

        if question.lower() == 'exit':
            print("Goodbye!")
            break

        answer = handle_question(question, index, cocktails)

        # Output the answer
        print("\nAnswer:")
        print(answer)
        print("\n")

if __name__ == "__main__":
    main()
