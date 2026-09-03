from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model="llama3",
    temperature=0
)


# ============================================================
# STEP 1: STRUCTURED DATA
# ============================================================

class ComplaintData(BaseModel):

    complaint: str = Field(
        description="The main customer complaint. "
                    "Return 'None' if there is no complaint."
    )

    product_or_feature: str = Field(
        description="The product, service, or feature mentioned. "
                    "Return 'Not provided' if not mentioned."
    )

    sentiment: str = Field(
        description="Customer sentiment: positive, negative, or neutral."
    )


# Tell Ollama to return this exact Pydantic structure
structured_llm = llm.with_structured_output(ComplaintData)


step1_prompt = PromptTemplate(
    template="""
You are a customer review analysis assistant.

Analyze the following customer review.

Rules:

- Extract the main complaint.
- Extract the product, service, or feature mentioned.
- Determine the sentiment as positive, negative, or neutral.
- If there is no complaint, return "None".
- If no product or feature is mentioned, return "Not provided".
- Use ONLY information present in the review.
- Do not invent information.

Customer Review:
{review_text}
""",
    input_variables=["review_text"]
)


step1_chain = step1_prompt | structured_llm


# ============================================================
# STEP 2: SUPPORT TICKET
# ============================================================

step2_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a support ticket writer.

Your job is to create a short and clear support ticket summary
for the internal support team.

IMPORTANT RULES:

- Use ONLY the structured complaint data provided.
- Do NOT use or assume information from the original customer review.
- Do NOT invent additional information.
- Keep the ticket professional and concise.
"""
        ),
        (
            "human",
            """Create a support ticket from the following structured data:

Complaint: {complaint}

Product or Feature: {product_or_feature}

Customer Sentiment: {sentiment}

Create a concise internal support ticket summary."""
        )
    ]
)


step2_parser = StrOutputParser()

step2_chain = step2_prompt | llm | step2_parser


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def customer_review_pipeline(review_text):

    # STEP 1
    structured_data = step1_chain.invoke(
        {
            "review_text": review_text
        }
    )

    # STEP 2
    ticket_summary = step2_chain.invoke(
        {
            "complaint": structured_data.complaint,
            "product_or_feature": structured_data.product_or_feature,
            "sentiment": structured_data.sentiment
        }
    )

    return structured_data, ticket_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    review_text = input("Enter customer review: ")

    structured_data, ticket_summary = customer_review_pipeline(
        review_text
    )

    print("\n========== STEP 1: STRUCTURED DATA ==========\n")

    print("Complaint:", structured_data.complaint)
    print("Product/Feature:", structured_data.product_or_feature)
    print("Sentiment:", structured_data.sentiment)

    print("\n========== STEP 2: SUPPORT TICKET ==========\n")

    print(ticket_summary)