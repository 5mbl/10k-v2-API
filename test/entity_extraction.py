import os
import re
import requests
from llama_index.llms.openai import OpenAI

# ✅ Set OpenAI API Key
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Initialize LLM
llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

# ✅ Define function to extract parameters
def extract_financial_params(query):
    """
    Extracts financial metric, company name, ticker symbol, and year from the user query.
    If either company name or ticker is missing, GPT should autofill it.
    """
    prompt = f"""
    Extract the following structured parameters from the user query:
    - 'metric': The financial metric requested (e.g., "gross revenue", "net income", "EBITDA").
    - 'company': The full company name (e.g., "Tesla", "Apple").
    - 'ticker': The correct stock ticker symbol (e.g., "TSLA" for Tesla, "AAPL" for Apple).
    - 'year': The year requested (e.g., "2022", "2023").

    If the company name is provided but the ticker is missing, infer the correct ticker.
    If the ticker is provided but the company name is missing, infer the full company name.
    
    Return ONLY a valid JSON object, formatted like this:
    {{"metric": "value", "company": "value", "ticker": "value", "year": "value"}}

    Query: "{query}"
    Answer:
    """

    response = llm.complete(prompt)

    try:
        extracted_data = eval(response.text.strip())  # Convert JSON-like response to dict
    except:
        extracted_data = {"metric": None, "company": None, "ticker": None, "year": None}

    return extracted_data

# ✅ Test Example
query = "Give me the gross revenue and net income of Tesla in 2022"
params = extract_financial_params(query)

print(f"🔍 Extracted Params: {params}")
