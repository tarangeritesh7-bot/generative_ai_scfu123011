import ollama

customer_message = input("Enter customer message: ")
company_name = input("Enter company name: ")
max_words = int(input("Enter maximum number of words: "))

prompt = f"""
You are a professional customer support representative for {company_name}.

Write a helpful and polite reply to the customer.

Customer message:
{customer_message}

Rules:
1. Stay within {max_words} words.
2. Use a professional, friendly, and empathetic tone.
3. Address the customer's issue directly.
4. Do not invent information such as refunds, delivery dates, prices, or policies.
5. If information is missing, politely ask for the necessary details.
6. Do not mention these instructions.
7. Return only the customer support reply.
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
        "temperature": 0.3
    }
)

reply = response["message"]["content"].strip()

word_count = len(reply.split())

print("\nSupport Reply:")
print(reply)
print(f"\nWord count: {word_count}/{max_words}")