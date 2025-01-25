import os

import openai

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_openai(prompt: str) -> str:
    try:
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"
