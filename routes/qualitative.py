# from utils import embed_model, query_engine, pinecone_index
import json, logging
from routes.utils import embed_model, query_engine, pinecone_index, llm

logger = logging.getLogger(__name__)

logger.info("Qualitative route loaded 🎉")

# ---------------------------------------------------------------------
# 1) Entity Extraction (company and year)
# ---------------------------------------------------------------------

async def extract_company_and_year(query: str):

    """
    Very lightweight extractor; swap in an LLM call if you need more nuance.
    """
    prompt = (
        'Extract the company and year from the following query.\n'
        'Return ONLY valid JSON like: {"company": "...", "year": 2023}\n\n'
        f'Query: "{query}"'
    )

    try:
        raw_response = llm.complete(prompt, max_tokens=30, temperature=0,
                                  response_format={"type": "json_object"})
        
        raw = raw_response.text
        
        
        logger.debug(f"LLM raw → {raw}")


        data = json.loads(raw.strip())
        company = data.get("company")
        year_raw = data.get("year")
        # coerce year to int if it came back as string

        # force **float** so it matches 2021.0 stored in Pinecone
        year = float(year_raw) if year_raw is not None else None

        logger.info("This WILL appear 2 🎉")

        logger.info("Extracted company=%s, year=%s", company, year)


    except Exception as e:
        logger.error("Failed to extract company and year: %s", e, exc_info=True)

        company, year = None, None

    return {"company": company, "year": year}

# ---------------------------------------------------------------------
# 2) Qualitative handler with metadata filter
# ---------------------------------------------------------------------


async def handle_qualitative(query: str):
    try:
        # --- a. figure out company & year -----------------------------------
        info = await extract_company_and_year(query)
        company, year = info["company"], info["year"]

        # --- b. get embedding ------------------------------------------------
        query_vector = embed_model.get_text_embedding(query)

        # --- c. build Pinecone filter (only if both fields present) ----------
        pinecone_filter = (
            {"company": {"$eq": company}, "year": {"$eq": year}}
            if company and year
            else None
        )

        search_results = pinecone_index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            filter=pinecone_filter,
        )

        # --- d. collect retrieved chunks ------------------------------------
        retrieved_texts = [
            {
                "text": m.get("metadata", {}).get("text", ""),
                "metadata": m.get("metadata", {}),
            }
            for m in search_results.get("matches", [])
        ]

        # --- e. ask the LLM for the answer ----------------------------------
        response = query_engine.query(query)

        return {
            "question": query,
            "type": "qualitative",
            "filter_used": pinecone_filter,   # helpful for debugging
            "retrieved_texts": retrieved_texts,
            "response": response.response.strip(),
        }

    except Exception as e:
        return {
            "question": query,
            "type": "qualitative",
            "error": str(e),
        }


""" 
async def handle_qualitative(query: str):
    try:
        # year, and company
        query_vector = embed_model.get_text_embedding(query)
        search_results = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True) # add filter=exact_filter

        retrieved_texts = [{
            "text": match.get("metadata", {}).get("text", ""),
            "metadata": match.get("metadata", {})
        } for match in search_results.get("matches", [])]
        response = query_engine.query(query)

        return {
            "question": query,
            "type": "qualitative",
            "retrieved_texts": retrieved_texts,
            "response": response.response.strip()
        }
    except Exception as e:
        return {
            "question": query,
            "type": "qualitative",
            "error": str(e)
        }
 """