import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).with_name(".env"), override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key or not api_key.startswith("gsk_"):
    raise RuntimeError(
        "Set a valid GROQ_API_KEY in .env or your environment."
    )

client = Groq(api_key=api_key)

patient_name = input("Enter patient name: ")

print("\nEnter patient consultation notes:")
patient_notes = input("> ")

prompt = f"""
You are a doctor's consultation summary assistant.

Convert the patient's raw consultation notes into a clean,
concise doctor's summary.

Patient Name: {patient_name}

Patient Notes:
{patient_notes}

Rules:
1. Use only information provided in the notes.
2. Do not invent any information.
3. Keep the summary concise and professional.
4. Always use exactly the same format.
5. Always include Symptoms, Diagnosis, and Recommendation.
6. If information is not provided, write "Not provided in notes".

Output format:

Patient Name: <patient name>

Symptoms:
- <symptom 1>
- <symptom 2>

Diagnosis:
<diagnosis>

Recommendation:
- <recommendation 1>
- <recommendation 2>
"""

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="openai/gpt-oss-120b",
    temperature=0,
)

print("\n========== DOCTOR SUMMARY ==========\n")
print(chat_completion.choices[0].message.content)