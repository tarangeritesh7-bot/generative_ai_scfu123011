import ollama
import json

print("Enter resume text.")
print("Type END on a new line when you are finished.\n")

resume_lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    resume_lines.append(line)

resume_text = "\n".join(resume_lines)

fields_input = input(
    "\nEnter fields to extract (comma separated): "
)

fields_to_extract = [
    field.strip()
    for field in fields_input.split(",")
    if field.strip()
]

fields = ", ".join(fields_to_extract)

prompt = f"""
Extract information from the resume below.

Resume:
{resume_text}

Requested fields:
{fields}

Rules:
1. Return ONLY valid JSON.
2. Return ONLY the requested fields.
3. Do not add extra fields.
4. Do not provide explanations.
5. Use the exact field names provided by the user.
6. If a requested field is not found, use null.
7. Return a JSON object.
"""

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    format="json",
    options={
        "temperature": 0
    }
)

result = response["message"]["content"]

try:
    data = json.loads(result)

    print("\nExtracted JSON:")
    print(json.dumps(data, indent=2))

except json.JSONDecodeError:
    print("Model did not return valid JSON.")
    print(result)