import json
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama


# Ollama model
llm = ChatOllama(
    model="mistral",
    temperature=0.1
)


class BugReport(BaseModel):
    steps_to_reproduce: list[str] = Field(
        description="Steps needed to reproduce the bug"
    )
    expected_behavior: str = Field(
        description="What should happen"
    )
    actual_behavior: str = Field(
        description="What actually happened"
    )
    severity: str = Field(
        description="Bug severity: Low, Medium, High, or Critical"
    )


parser = PydanticOutputParser(pydantic_object=BugReport)


# ---------------- STEP 1 ----------------

prompt1 = PromptTemplate(
    template="""Convert the bug report into the required JSON object.

BUG REPORT:
{bug_report_text}

Your response MUST contain ONLY this JSON object.
Do not write anything before or after the JSON.

Required format:

{{
    "steps_to_reproduce": [],
    "expected_behavior": "",
    "actual_behavior": "",
    "severity": ""
}}

Rules:
- Extract only information from the bug report.
- Do not invent information.
- Use an empty list if reproduction steps are missing.
- Use "Not provided" if expected behavior is missing.
- Use "Not provided" if actual behavior is missing.
- Use "Low" if severity is not mentioned and there is not enough information to determine it.
- Severity must be Low, Medium, High, or Critical.
- Do not include explanations.
- Do not include markdown.
- Do not return a JSON schema.

{format_instructions}
""",
    input_variables=["bug_report_text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Get raw response first
step1_raw = prompt1 | llm | StrOutputParser()


def extract_json(text):
    """Extract JSON object from the model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


def create_structured_report(bug_report):
    raw_output = step1_raw.invoke({
        "bug_report_text": bug_report
    })

    data = extract_json(raw_output)

    return BugReport.model_validate(data)

# ---------------- STEP 2 ----------------

prompt2 = PromptTemplate(
    template="""You are a software QA reviewer.

Review the structured bug report below.

Steps to reproduce:
{steps_to_reproduce}

Expected behavior:
{expected_behavior}

Actual behavior:
{actual_behavior}

Severity:
{severity}

Identify missing or unclear information.

Check:
- Missing reproduction steps
- Missing expected behavior
- Missing actual behavior
- Missing or unclear severity
- Other important information needed to reproduce or fix the bug

Return a short list of specific gaps.

If everything important is available, return:
No major gaps found.
""",
    input_variables=[
        "steps_to_reproduce",
        "expected_behavior",
        "actual_behavior",
        "severity"
    ]
)

step2 = prompt2 | llm | StrOutputParser()


# ---------------- STEP 3 ----------------

prompt3 = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior software engineer.
Create a practical and prioritized fix plan for a development team."""
    ),
    (
        "human",
        """Structured bug report:

Steps to reproduce:
{steps_to_reproduce}

Expected behavior:
{expected_behavior}

Actual behavior:
{actual_behavior}

Severity:
{severity}


Flagged gaps:
{gaps}


Create a prioritized fix plan.

Include:
1. Priority
2. Action
3. Reason
4. Testing or validation

Rules:
- Use only the information provided.
- Do not invent technical details.
- Do not assume a programming language or framework.
- Mention missing information that needs to be collected.
- Keep the plan concise.
"""
    )
])

step3 = prompt3 | llm | StrOutputParser()


# ---------------- MAIN PROGRAM ----------------

print("=== Bug Report to Fix Plan Pipeline ===")

bug_report_text = input("\nEnter messy bug report:\n")


# Step 1
structured_report = create_structured_report(
    bug_report_text
)


# Step 2
gaps = step2.invoke(
    structured_report.model_dump()
)


# Step 3
fix_plan = step3.invoke({
    **structured_report.model_dump(),
    "gaps": gaps
})


# ---------------- OUTPUT ----------------

print("\n========== STEP 1: STRUCTURED BUG REPORT ==========\n")
print(structured_report.model_dump_json(indent=2))

print("\n========== STEP 2: FLAGGED GAPS ==========\n")
print(gaps)

print("\n========== STEP 3: PRIORITIZED FIX PLAN ==========\n")
print(fix_plan)