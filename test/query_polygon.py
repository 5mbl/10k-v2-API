import requests

# ✅ Polygon API Key
POLYGON_API_KEY = "MVanSy_ESI_yPzeDKuolZigyuK6jswGW"  # 🔥 Replace with your actual API key

# ✅ Function to fetch financials for a specific year
def get_financials(ticker, year):
    url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=annual&filing_date.gte={year}-01-01&filing_date.lte={year}-12-31&limit=1&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    # ✅ Check if data exists
    if "results" not in data or len(data["results"]) == 0:
        print(f"❌ No financial data found for {ticker} in {year}")
        return None

    # ✅ Extract financials
    financials = data["results"][0]

    # ✅ Extract important metrics
    report_date = financials.get("period_of_report", "Unknown")
    revenue = financials.get("financials", {}).get("income_statement", {}).get("revenues", {}).get("value", None)
    net_income = financials.get("financials", {}).get("income_statement", {}).get("net_income", {}).get("value", None)
    total_assets = financials.get("financials", {}).get("balance_sheet", {}).get("assets", {}).get("value", None)
    total_liabilities = financials.get("financials", {}).get("balance_sheet", {}).get("liabilities", {}).get("value", None)
    gross_profit = financials.get("financials", {}).get("income_statement", {}).get("gross_profit", {}).get("value", None)


    # ✅ Print results
    print(f"📆 Financials for {ticker} in {year} (Reported on {report_date})")
    print(f"💰 Gross Revenue: ${revenue:,}" if revenue else "❌ Revenue data not available")
    print(f"📉 Net Income: ${net_income:,}" if net_income else "❌ Net Income data not available")
    print(f"🏦 Total Assets: ${total_assets:,}" if total_assets else "❌ Total Assets data not available")
    print(f"💳 Total Liabilities: ${total_liabilities:,}" if total_liabilities else "❌ Total Liabilities data not available")
    print(f"💸 Gross Profit: ${gross_profit:,}" if gross_profit else "❌ Gross Profit: data not available")


    return {
        "year": year,
        "report_date": report_date,
        "revenue": revenue,
        "net_income": net_income,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "gross_profit": gross_profit

    }

# 🛠 Example Usage
if __name__ == "__main__":
    get_financials("TSLA", 2022)  # Fetch Tesla's annual financials for 2022
