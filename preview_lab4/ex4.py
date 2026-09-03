from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama



llm = ChatOllama(
    model="llama3",
    temperature=0.2
)



class ProductPitch(BaseModel):
    problem: str = Field(description="Problem the product solves")
    solution: str = Field(description="How the product solves the problem")
    target_user: str = Field(description="Target user of the product")


parser = PydanticOutputParser(pydantic_object=ProductPitch)



prompt1 = PromptTemplate(
    template="""You are given a product idea.

Product idea:
{product_idea}

Extract the following information:
1. The problem being solved
2. The proposed solution
3. The target user

Return only valid JSON in this format:

{{
    "problem": "problem description",
    "solution": "solution description",
    "target_user": "target user"
}}

Important:
- Fill in actual values.
- Do not return a JSON schema.
- Do not include properties, title, description, or type.
- Do not add extra fields.
- Do not use markdown.
- Do not invent statistics or facts.

{format_instructions}
""",
    input_variables=["product_idea"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

step1 = prompt1 | llm | parser





prompt2 = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an investor pitch writer.
Write a short and convincing pitch using only the
structured product information provided by the user."""
    ),
    (
        "human",
        """Product information:

Problem: {problem}
Solution: {solution}
Target User: {target_user}

Write one short investor-style pitch paragraph.

Rules:
- Use only the information provided above.
- Do not refer to the original product idea.
- Do not make up statistics, revenue, market size, or traction.
- Keep the pitch concise and professional.
"""
    )
])

step2 = prompt2 | llm | StrOutputParser()




print("=== Product Idea to Pitch Pipeline ===")

product_idea = input("\nEnter one-line product idea: ")



structured_pitch = step1.invoke({
    "product_idea": product_idea
})



pitch_paragraph = step2.invoke(
    structured_pitch.model_dump()
)



print("\n========== STEP 1: STRUCTURED PITCH ==========\n")
print(structured_pitch.model_dump_json(indent=2))

print("\n========== STEP 2: INVESTOR PITCH ==========\n")
print(pitch_paragraph)