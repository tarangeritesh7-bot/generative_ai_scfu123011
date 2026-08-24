import ollama

review_text = input("Enter customer review: ")

prompt = f"""
Classify the following customer review into exactly ONE category.

Categories:
POSITIVE - The customer is satisfied or happy.
NEGATIVE - The customer is unhappy or dissatisfied.
COMPLAINT - The customer reports a specific problem or issue.
SUGGESTION - The customer recommends an improvement or new feature.

Rules:
- Return ONLY the category name.
- Do not provide an explanation.
- Do not return multiple categories.
- Use only one of these categories:
  POSITIVE, NEGATIVE, COMPLAINT, SUGGESTION.

Customer review:
{review_text}
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
        "temperature": 0
    }
)

category = response["message"]["content"].strip()

print("\nCategory:", category)