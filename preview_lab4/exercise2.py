import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="moonshotai/kimi-k3",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2
)


class Claims(BaseModel):
    claims: list[str] = Field(
        description="List of factual claims extracted from the article"
    )


parser1 = PydanticOutputParser(pydantic_object=Claims)

prompt1 = PromptTemplate(
    template="""Extract the core factual claims from this news article.

Article:
{article_text}

{format_instructions}

Rules:
- Extract only claims explicitly stated in the article.
- Do not add outside information.
- Do not guess or invent facts.
- Keep each claim concise.
""",
    input_variables=["article_text"],
    partial_variables={
        "format_instructions": parser1.get_format_instructions()
    }
)

step1 = prompt1 | llm | parser1


prompt2 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a news fact-card writer. Create a concise fact card "
        "using ONLY the claims provided to you."
    ),
    (
        "human",
        """Extracted claims:

{claims}

Create a fact card containing:

Headline: <short headline>

- <bullet point 1>
- <bullet point 2>
- <bullet point 3>

Source confidence: <brief confidence note>

Rules:
- Use only the extracted claims.
- Do not add outside information.
- Do not invent facts.
- Use exactly 3 bullet points.
- If fewer than 3 claims are available, do not invent additional claims.
"""
    )
])

step2 = prompt2 | llm | StrOutputParser()


print("=== News Article to Fact Card Pipeline ===")

article_text = input("\nEnter news article:\n")

# STEP 1
claims = step1.invoke({
    "article_text": article_text
})

# STEP 2
fact_card = step2.invoke({
    "claims": "\n".join(f"- {claim}" for claim in claims.claims)
})


print("\n========== STEP 1: EXTRACTED CLAIMS ==========\n")

for i, claim in enumerate(claims.claims, 1):
    print(f"{i}. {claim}")

print("\n========== STEP 2: FACT CARD ==========\n")
print(fact_card)