import json, requests, logging
# from utils import llm, POLYGON_API_KEY
from routes.utils import llm, POLYGON_API_KEY
import re

logger = logging.getLogger(__name__)

logger.info("Quantitative route module initialized")

async def handle_quantitative(query: str):
    """
    Process quantitative financial queries by extracting parameters, fetching data from Polygon API,
    and extracting specific metric values using LLM.
    """
    logger.info("Processing quantitative query: %s", query[:100] + "..." if len(query) > 100 else query)
    
    try:
        # Step 1: Extract structured parameters from natural language query
        logger.debug("Step 1: Extracting financial parameters from query")
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
        logger.debug("LLM extraction response: %s", extracted)

        # Parse JSON with fallback regex extraction if initial parsing fails
        try:
            params = json.loads(extracted)
            logger.debug("Parameters extracted successfully: %s", params)
        except json.JSONDecodeError:
            logger.warning("Initial JSON parsing failed, attempting fallback extraction")
            # Fallback: attempt to extract the first JSON object from the LLM output
            match = re.search(r'\{.*?\}', extracted, re.DOTALL)
            if match:
                try:
                    params = json.loads(match.group(0))
                    logger.debug("Fallback JSON parsing successful: %s", params)
                except Exception as inner:
                    logger.error("Fallback JSON parsing failed: %s", inner)
                    raise ValueError(f"Fallback JSON parsing failed: {inner}\nRaw extracted:\n{extracted}")
            else:
                logger.error("No JSON object found in LLM output")
                raise ValueError(f"Could not parse JSON from LLM output:\n{extracted}")

        # Validate required parameters
        if not params.get("ticker") or not params.get("year"):
            logger.error("Missing required parameters - ticker: %s, year: %s", params.get("ticker"), params.get("year"))
            return {"question": query, "type": "quantitative", "error": "Missing ticker or year"}

        logger.info("Extracted parameters - company: %s, ticker: %s, year: %s, metric: %s", 
                   params.get("company"), params.get("ticker"), params.get("year"), params.get("metric"))

        # Step 2: Fetch financial data from Polygon API for the specified ticker and year
        logger.debug("Step 2: Fetching financial data from Polygon API")
        url = f"https://api.polygon.io/vX/reference/financials?ticker={params['ticker']}&timeframe=annual&filing_date.gte={params['year']}-01-01&filing_date.lte={params['year']}-12-31&limit=1&apiKey={POLYGON_API_KEY}"
        res = requests.get(url).json()

        if "results" not in res or not res["results"]:
            logger.error("No financial data found for ticker %s in year %s", params['ticker'], params['year'])
            return {"question": query, "type": "quantitative", "error": "Financial data not found"}

        logger.debug("Polygon API returned %d results", len(res["results"]))
        polygon_data = res["results"][0]
        
        # Step 3: Use LLM to extract the specific metric value from the financial data
        logger.debug("Step 3: Extracting specific metric value from financial data")
        value_prompt = f"""
        Given this JSON:
        {json.dumps(polygon_data, indent=2)}

        Return only the numeric value of the '{params['metric']}' field. Do not include any explanation, label, or formatting — just the number.
        """
        value = llm.complete(value_prompt).text.strip()
        logger.debug("Extracted metric value: %s", value)

        logger.info("Quantitative query processed successfully - %s %s %s: %s", 
                   params.get("company"), params.get("metric"), params.get("year"), value)
        
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
        logger.error("Quantitative query processing failed: %s", e, exc_info=True)
        return {"question": query, "type": "quantitative", "error": str(e)}
