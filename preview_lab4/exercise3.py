import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("NVIDIA_API_KEY")

url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


def ask(prompt, temperature=0.2):
    payload = {
        "model": "minimaxai/minimax-m3",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 4096,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("API Error:")
        print(response.text)
        raise SystemExit

    return response.json()["choices"][0]["message"]["content"]


print("=== Meeting Transcript to Action Items Pipeline ===")

print("\nEnter meeting transcript.")
print("Type END when finished.\n")

lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    lines.append(line)

transcript_text = "\n".join(lines)


# STEP 1 - Discussion Summary

step1_prompt = f"""
Read the meeting transcript and summarize what was discussed.

TRANSCRIPT:
{transcript_text}

Rules:
1. Include only important discussion points.
2. Do not invent information.
3. Keep the summary clear and concise.
4. Return only the discussion summary.
"""

discussion_summary = ask(step1_prompt)


# STEP 2 - Action Items

step2_prompt = f"""
Identify action items from the discussion summary below.

DISCUSSION SUMMARY:
{discussion_summary}

Return ONLY valid JSON in this format:

[
  {{
    "task": "",
    "owner": "",
    "deadline": "",
    "flag": ""
  }}
]

Rules:
1. Identify only actual action items.
2. If owner is missing, use "Missing".
3. If deadline is missing, use "Missing".
4. If owner or deadline is missing, set flag to "Needs clarification".
5. Otherwise set flag to "Complete".
6. Do not invent owners or deadlines.
7. Return JSON only.
"""

step2_result = ask(step2_prompt, temperature=0)

step2_result = (
    step2_result
    .replace("```json", "")
    .replace("```", "")
    .strip()
)

try:
    action_items = json.loads(step2_result)
except json.JSONDecodeError:
    print("\nStep 2 returned invalid JSON:")
    print(step2_result)
    raise SystemExit


# STEP 3 - Structured Task Table

step3_prompt = f"""
Format the following action items as a Markdown task table.

ACTION ITEMS:
{json.dumps(action_items, indent=2)}

Use exactly these columns:

| Task | Owner | Deadline | Status |

Rules:
1. Do not add new tasks.
2. Keep missing owner or deadline as "Missing".
3. Use the flag value as Status.
4. Return only the Markdown table.
"""

task_table = ask(step3_prompt, temperature=0)


# OUTPUT

print("\n========== STEP 1: DISCUSSION SUMMARY ==========\n")
print(discussion_summary.strip())

print("\n========== STEP 2: FLAGGED ACTION ITEMS ==========\n")
print(json.dumps(action_items, indent=2))

print("\n========== STEP 3: STRUCTURED TASK TABLE ==========\n")
print(task_table.strip())