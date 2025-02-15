# Cocktail Advisor

## Overview

Cocktail Advisor is an interactive CLI tool designed to recommend cocktails based on user preferences. The system leverages a FAISS index for similarity search and incorporates both natural language processing (NLP) for handling user queries and a large language model (LLM) for answering general questions.

## Features

- **Cocktail Recommendations**\*: Suggests cocktails based on user-specified ingredients.\*
- **Similarity Search**\*: Finds cocktails similar to a given one.\*
- **Interactive CLI**\*: Users can interact with the system in real time.\*
- **LLM Integration**\*: Provides informative answers to general questions.\*

## Project S*tructure*

```
project/
│
├── src/
│   ├── database.py         # Handles FAISS index creation and loading cocktail data
│   ├── embeddings.py       # Provides functions for embedding and FAISS index management
│   ├── rag.py              # Contains recommendation and query-processing logic
│   ├── llm.py              # Integrates with a language model for answering general questions
│
├── data/
│   ├── final_cocktails.csv  # Input dataset with cocktail details
│   ├── cocktail_index.faiss # FAISS index for cocktail embeddings
│
└── main.py                 # Entry point for the Cocktail Advisor CLI
```

## Prerequisites

- Python 3.8+
- Required Python libraries (see below)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SashaKisliy/cocktail-advisor-chat.git
   cd cocktail-advisor-chat
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset (`final_cocktails.csv`) is present in the `data/` directory.

## Usage

Run the application using:

```bash
python main.py
```

### Example Interaction

1. Start the CLI:
   ```
   Welcome to the Cocktail Advisor Chat CLI!
   Ask a question (or 'exit' to quit):
   ```
2. Ask for recommendations:
   ```
   Ask a question (or 'exit' to quit): Recommend 5 cocktails containing lemon, vodka
   Answer:
   Recommended cocktails: Margarita, Lemon Drop, Vodka Collins, etc.
   ```
3. Exit the application:
   ```
   Ask a question (or 'exit' to quit): exit
   Goodbye!
   ```

## How It Works

### Recommendation Workflow

1. **Load or Prepare FAISS Index**: If the FAISS index exists, it is loaded; otherwise, it is created using the cocktail dataset.
2. **Process User Query**: Queries are parsed to extract ingredients or other preferences.
3. **Recommendation Engine**: Based on the query, the system performs a vector similarity search using the FAISS index.
4. **LLM Integration**: If the query is not for recommendations, the system passes it to the integrated LLM for a response.

### Data

- Cocktails Dataset - https://www.kaggle.com/datasets/aadyasingh55/cocktails

## Dependencies

- FAISS
- NumPy
- Pandas
- OpenAI API (or another LLM provider)

## Future Improvements

- Expand the dataset with more cocktails.
- Improve the natural language processing capabilities for better query understanding.
- Add a web or GUI interface.



