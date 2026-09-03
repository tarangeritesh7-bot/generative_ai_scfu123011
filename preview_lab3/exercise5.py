import ollama

sentence = input("Enter sentence: ")
target_language = input("Enter target language: ")
formality = input("Enter formality level (formal/informal): ")

prompt = f"""
Translate the following sentence into {target_language}.

Sentence:
{sentence}

Formality level:
{formality}

Rules:
1. Return only the translated sentence.
2. Do not provide explanations.
3. Do not include quotation marks.
4. Preserve the original meaning.
5. Match the requested formality level exactly.
6. Do not add information that is not present in the original sentence.
"""

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0.2
    }
)

translation = response["message"]["content"].strip()

print("\nTranslation:")
print(translation)