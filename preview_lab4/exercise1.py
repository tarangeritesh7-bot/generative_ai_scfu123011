import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
key = os.getenv("NVIDIA_API_KEY")

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def ask(prompt):
    data = {
        "model": "minimaxai/minimax-m3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 4096
    }
    r = requests.post(url, headers=headers, json=data)
    if r.status_code != 200:
        print(r.text)
        exit()
    return r.json()["choices"][0]["message"]["content"]


def get_input(title):
    print(f"\nEnter {title}. Type END when finished.")
    lines = []
    while True:
        x = input()
        if x.strip() == "END":
            return "\n".join(lines)
        lines.append(x)


print("=== Job Posting to Candidate Outreach Pipeline ===")

job = get_input("job posting")
candidate = get_input("candidate profile")



prompt1 = f"""
Extract key requirements from this job posting:

{job}

Return ONLY JSON:
{{
"job_title":"",
"required_skills":[],
"preferred_skills":[],
"experience":"",
"education":"",
"location":"",
"employment_type":""
}}

Do not invent information. Use empty values when missing.
"""

result1 = ask(prompt1).replace("```json", "").replace("```", "").strip()
requirements = json.loads(result1)



prompt2 = f"""
Write a short, professional and personalized outreach message.

JOB REQUIREMENTS:
{json.dumps(requirements)}

CANDIDATE:
{candidate}

Use only the structured job requirements, not the original job posting.
Mention matching skills or experience.
Do not invent information.
Return only the message.
"""

message = ask(prompt2)

print("\n========== STEP 1 ==========\n")
print(json.dumps(requirements, indent=2))

print("\n========== STEP 2 ==========\n")
print(message.strip())