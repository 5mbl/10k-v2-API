import json, requests
# from utils import llm, POLYGON_API_KEY
from routes.utils import llm, POLYGON_API_KEY
import re


async def handle_quantitative(query: str):
    try:
        extraction_prompt = f"""
        Extract the following structured parameters from the user query:
        - 'metric': The financial metric requested (e.g., revenue, gross_profit, liabilities, cashflow, net_income).
        - 'company': Full company name.
        - 'ticker': Stock ticker symbol (e.g., TSLA for Tesla, AAPL for Apple).
        - 'year': Year requested.

        For common companies, please include their ticker symbol even if not explicitly mentioned:
        - Tesla → TSLA
        - Apple → AAPL
        - Microsoft → MSFT
        - Amazon → AMZN
        - Google → GOOGL

        Return as valid JSON.
        Query: "{query}"
        """
        extracted = llm.complete(extraction_prompt).text.strip()
        #params = json.loads(extracted)

        try:
            params = json.loads(extracted)
        except json.JSONDecodeError:
            # Fallback: attempt to extract the first JSON object from the LLM output
            match = re.search(r'\{.*?\}', extracted, re.DOTALL)
            if match:
                try:
                    params = json.loads(match.group(0))
                except Exception as inner:
                    raise ValueError(f"Fallback JSON parsing failed: {inner}\nRaw extracted:\n{extracted}")
            else:
                raise ValueError(f"Could not parse JSON from LLM output:\n{extracted}")


        if not params.get("ticker") or not params.get("year"):
            return {"question": query, "type": "quantitative", "error": "Missing ticker or year"}

        url = f"https://api.polygon.io/vX/reference/financials?ticker={params['ticker']}&timeframe=annual&filing_date.gte={params['year']}-01-01&filing_date.lte={params['year']}-12-31&limit=1&apiKey={POLYGON_API_KEY}"
        res = requests.get(url).json()

        if "results" not in res or not res["results"]:
            return {"question": query, "type": "quantitative", "error": "Financial data not found"}

        polygon_data = res["results"][0]
        value_prompt = f"""
        Given this JSON:
        {json.dumps(polygon_data, indent=2)}

        Return only the numeric value of the '{params['metric']}' field. Do not include any explanation, label, or formatting — just the number.
        """
        value = llm.complete(value_prompt).text.strip()

        return {
            "question": query,
            "type": "quantitative",
            "company": params.get("company"),
            "ticker": params.get("ticker"),
            "year": params.get("year"),
            "metric": params.get("metric"),
            "value": value
        }
    except Exception as e:
        return {"question": query, "type": "quantitative", "error": str(e)}
