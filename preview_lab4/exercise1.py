import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise RuntimeError("NVIDIA_API_KEY is missing from .env")

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

MODEL = "minimaxai/minimax-m3"


def call_model(prompt, temperature=0):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 4096,
        "stream": False
    }

    response = requests.post(
        invoke_url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("API Error:")
        print(response.text)
        exit()

    result = response.json()

    return result["choices"][0]["message"]["content"]


print("=== Job Posting to Candidate Outreach Pipeline ===")

print("\nEnter job posting.")
print("Type END when finished.\n")

job_lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    job_lines.append(line)

job_posting_text = "\n".join(job_lines)


print("\nEnter candidate profile.")
print("Type END when finished.\n")

candidate_lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    candidate_lines.append(line)

candidate_profile = "\n".join(candidate_lines)


# STEP 1

step1_prompt = f"""
Extract the key requirements from the following job posting.

JOB POSTING:
{job_posting_text}

Return ONLY valid JSON with exactly these fields:

{{
    "job_title": "",
    "required_skills": [],
    "preferred_skills": [],
    "experience": "",
    "education": "",
    "location": "",
    "employment_type": ""
}}

Rules:
1. Extract only information explicitly present in the job posting.
2. Do not invent information.
3. Use an empty string when information is missing.
4. Use an empty list when no skills are provided.
5. Return JSON only.
"""

step1_result = call_model(step1_prompt, temperature=0)

step1_result = (
    step1_result
    .replace("```json", "")
    .replace("```", "")
    .strip()
)

try:
    structured_requirements = json.loads(step1_result)

except json.JSONDecodeError:

    print("\nStep 1 returned invalid JSON:")
    print(step1_result)

    exit()


# STEP 2

step2_prompt = f"""
Create a personalized outreach message for the candidate.

STRUCTURED JOB REQUIREMENTS:
{json.dumps(structured_requirements, indent=2)}

CANDIDATE PROFILE:
{candidate_profile}

Important rules:

1. Use ONLY the structured job requirements above.
2. DO NOT use the original job posting.
3. Personalize the message using the candidate profile.
4. Mention matching skills or experience.
5. Do not invent candidate qualifications.
6. Keep the message professional, friendly, and concise.
7. Return ONLY the outreach message.
"""

step2_result = call_model(step2_prompt, temperature=0.3)


print("\n========== STEP 1: STRUCTURED REQUIREMENTS ==========\n")

print(json.dumps(
    structured_requirements,
    indent=2
))


print("\n========== STEP 2: PERSONALIZED OUTREACH ==========\n")

print(step2_result.strip())