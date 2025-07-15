from llama_index.llms.openai import OpenAI
import os

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Initialize LLM with OpenAI GPT-4
""" llm = OpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)  # Set temperature=0 for deterministic results
agent = OpenAIAgent.from_defaults(llm=llm) """

llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

#chat_engine = index.as_chat_engine(chat_mode="openai", llm=llm, verbose=True)


def classify_query(query):
    """
    Classifies a user query into:
    - 'quantitative' (financial metrics)
    - 'qualitative' (textual analysis)
    - 'hybrid' (both types of data)
    """
    classification_prompt = f"""
    Classify the following query into one of these categories:
    - 'quantitative': If it asks about revenue, net income, stock price, earnings, financial ratios, or other numerical financial data.
    - 'qualitative': If it asks about company strategy, risks, business operations, leadership, or general non-financial company analysis.
    - 'hybrid': If the query requires both financial and textual analysis.

    Query: "{query}"
    
    Respond with only one word: 'quantitative', 'qualitative', or 'hybrid'. Do not provide any explanation.
    """


    #response = chat_engine.chat(classification_prompt)
    response = llm.complete(classification_prompt)

    # Ensure clean classification response
    query_type = response.text.strip().lower()
    return query_type

# 🛠 Example Usage
if __name__ == "__main__":
    test_queries = [
        "What was Apple's revenue in 2023?",  # quantitative
        "Tell me about Tesla's business strategy.",  # qualitative
        "What are the risk factors in Amazon's latest 10-K?",  # qualitative
        "How much net income did Microsoft report last year?",  # quantitative
        "Summarize Tesla’s revenue and key business risks from its 10-K."  # hybrid
    ]
    
    for query in test_queries:
        result = classify_query(query)
        print(f"🔍 Query: {query}")
        print(f"🧠 Classification: {result.upper()}\n")
