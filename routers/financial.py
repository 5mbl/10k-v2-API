from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
import requests
import os
import json

# ✅ Load API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not OPENAI_API_KEY or not POLYGON_API_KEY:
    raise ValueError("Missing API keys")

# ✅ Initialize LLM
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# ✅ Create FastAPI Router
router = APIRouter()

# ✅ Define Request Model
class FinancialQuery(BaseModel):
    query: str

# ✅ Function to extract financial parameters
def extract_financial_params(query):
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
        extracted_data = json.loads(response.text.strip())
    except json.JSONDecodeError:
        extracted_data = {"metric": None, "company": None, "ticker": None, "year": None}

    return extracted_data

# ✅ Function to fetch financial data from Polygon
def get_financials(ticker, year):
    url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=annual&filing_date.gte={year}-01-01&filing_date.lte={year}-12-31&limit=1&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return None

    return data["results"][0]

# ✅ Function to extract the requested metric from financial data
def extract_metric_from_data(financial_data, metric):
    prompt = f"""
    Here is the financial data from Polygon.io in JSON format:

    {json.dumps(financial_data, indent=2)}

    Extract the value for the requested metric: "{metric}".
    
    If the metric is not available, return "Metric not found".
    
    Answer:
    """

    response = llm.complete(prompt)
    return response.text.strip()

# ✅ Define the API Endpoint
@router.post("/financial/")
async def get_financial_metric(query: FinancialQuery):
    # Step 1: Extract Financial Parameters
    params = extract_financial_params(query.query)

    if not params["ticker"] or not params["year"]:
        raise HTTPException(status_code=400, detail="Unable to extract valid ticker or year from query")

    # Step 2: Fetch Financial Data
    financial_data = get_financials(params["ticker"], params["year"])

    if not financial_data:
        raise HTTPException(status_code=404, detail=f"No financial data found for {params['ticker']} in {params['year']}")

    # Step 3: Extract Requested Metric
    extracted_value = extract_metric_from_data(financial_data, params["metric"])

    return {
        "company": params["company"],
        "ticker": params["ticker"],
        "year": params["year"],
        "metric": params["metric"],
        "value": extracted_value
    }
