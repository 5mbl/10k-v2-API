import os
import requests
import json
from llama_index.llms.openai import OpenAI

# ✅ Set API Keys
OPENAI_API_KEY = "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"
POLYGON_API_KEY = "MVanSy_ESI_yPzeDKuolZigyuK6jswGW"  # 🔥 Replace with your actual Polygon API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Initialize LLM
llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

# ✅ Function to extract financial parameters
def extract_financial_params(query):
    """
    Extracts financial metric, company name, ticker symbol, and year from the user query.
    Uses GPT to autofill missing details.
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
        extracted_data = json.loads(response.text.strip())  # Convert JSON response to dict
    except:
        extracted_data = {"metric": None, "company": None, "ticker": None, "year": None}

    return extracted_data

# ✅ Function to fetch financials from Polygon
def get_financials(ticker, year):
    url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=annual&filing_date.gte={year}-01-01&filing_date.lte={year}-12-31&limit=1&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        print(f"❌ No financial data found for {ticker} in {year}")
        return None

    return data["results"][0]  # Return full financial data for RAG processing

# ✅ Function to extract requested metric using GPT
def extract_metric_from_data(financial_data, metric):
    """
    Uses GPT to extract the requested financial metric from the Polygon response.
    """
    prompt = f"""
    Here is the financial data from Polygon.io in JSON format:

    {json.dumps(financial_data, indent=2)}

    Extract the value for the requested metric: "{metric}".
    
    If the metric is not available, return "Metric not found".
    
    Answer:
    """

    response = llm.complete(prompt)
    return response.text.strip()

# 🛠 Example Usage
if __name__ == "__main__":
    user_query = "Give me the gross revenue and net income of Tesla in 2022"
    
    # 🔍 Extract financial parameters
    params = extract_financial_params(user_query)
    print(f"🔍 Extracted Params: {params}")

    if params["ticker"] and params["year"]:
        financial_data = get_financials(params["ticker"], params["year"])
        
        if financial_data:
            # 🔹 Extract requested metric using GPT
            extracted_value = extract_metric_from_data(financial_data, params["metric"])
            print(f"📊 Extracted {params['metric']}: {extracted_value}")
