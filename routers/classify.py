from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
import os

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from environment variables")

llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

router = APIRouter()

# Request model
class QueryInput(BaseModel):
    query: str

@router.post("/classify/")
async def classify(query_input: QueryInput):
    query = query_input.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    classification_prompt = f"""
    Classify the following query into one of these categories:
    - 'quantitative': If it asks about revenue, net income, stock price, earnings, financial ratios, or other numerical financial data.
    - 'qualitative': If it asks about company strategy, risks, business operations, leadership, or general non-financial company analysis.
    - 'hybrid': If the query requires both financial and textual analysis.

    Query: "{query}"
    
    Respond with only one word: 'quantitative', 'qualitative', or 'hybrid'. Do not provide any explanation.
    """

    try:
        response = llm.complete(classification_prompt)
        query_type = response.text.strip().lower()
        return {"query": query, "classification": query_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
