# from utils import embed_model, query_engine, pinecone_index
import json, logging
from routes.utils import embed_model, query_engine, pinecone_index, llm

logger = logging.getLogger(__name__)

logger.info("Qualitative route module initialized")

# ---------------------------------------------------------------------
# 1) Entity Extraction (company and year)
# ---------------------------------------------------------------------

async def extract_company_and_year(query: str):


    logger.info("Starting entity extraction for query: %s", query[:100] + "..." if len(query) > 100 else query)
    
    prompt = (
        'Extract the company and year from the following query.\n'
        'Return ONLY valid JSON like: {"company": "...", "year": 2023}\n\n'
        f'Query: "{query}"'
    )

    try:
        raw_response = llm.complete(prompt, max_tokens=30, temperature=0,
                                  response_format={"type": "json_object"})
        
        raw = raw_response.text
        
        logger.debug("LLM response received: %s", raw)

        data = json.loads(raw.strip())
        company = data.get("company")
        year_raw = data.get("year")
        # coerce year to int if it came back as string

        # force **float** so it matches 2021.0 stored in Pinecone
        year = float(year_raw) if year_raw is not None else None

        logger.info("Entity extraction completed successfully - company: %s, year: %s", company, year)

    except Exception as e:
        logger.error("Entity extraction failed: %s", e, exc_info=True)
        company, year = None, None

    return {"company": company, "year": year}

# ---------------------------------------------------------------------
# 2) Qualitative handler with metadata filter
# ---------------------------------------------------------------------


async def handle_qualitative(query: str):
    logger.info("Processing qualitative query: %s", query[:100] + "..." if len(query) > 100 else query)
    
    try:
        # --- a. figure out company & year -----------------------------------
        logger.debug("Step 1: Extracting company and year from query")
        info = await extract_company_and_year(query)
        company, year = info["company"], info["year"]

        # --- b. get embedding ------------------------------------------------
        logger.debug("Step 2: Generating query embedding")
        query_vector = embed_model.get_text_embedding(query)

        # --- c. build Pinecone filter (only if both fields present) ----------
        pinecone_filter = (
            {"company": {"$eq": company}, "year": {"$eq": year}}
            if company and year
            else None
        )
        logger.debug("Step 3: Pinecone filter created: %s", pinecone_filter)

        logger.debug("Step 4: Querying Pinecone index")
        search_results = pinecone_index.query(
            vector=query_vector,
            top_k=10,
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
        logger.debug("Step 5: Retrieved %d text chunks from Pinecone", len(retrieved_texts))

        # --- e. ask the LLM for the answer ----------------------------------
        logger.debug("Step 6: Generating final response using query engine")
        response = query_engine.query(query)

        logger.info("Qualitative query processed successfully - retrieved %d chunks", len(retrieved_texts))
        return {
            "question": query,
            "type": "qualitative",
            "filter_used": pinecone_filter,   # helpful for debugging
            "retrieved_texts": retrieved_texts,
            "response": response.response.strip(),
        }

    except Exception as e:
        logger.error("Qualitative query processing failed: %s", e, exc_info=True)
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