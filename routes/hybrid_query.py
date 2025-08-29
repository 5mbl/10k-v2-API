from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json

# from utils import split_into_subquestions, classify_subquestion, merge_responses
from routes.utils import split_into_subquestions, classify_subquestion, merge_responses

# from qualitative import handle_qualitative
from routes.qualitative import handle_qualitative

# from quantitative import handle_quantitative
from routes.quantitative import handle_quantitative


router = APIRouter()

# Pydantic model for request validation
# Ensures the incoming query is properly formatted as a string
class HybridQuery(BaseModel):
    query: str

# starting point for queries that may contain both qualitative and quantitative components
@router.post("/hybrid-query")
async def hybrid_query(input: HybridQuery):
    # Extract the query text from the validated input
    query_text = input.query

    # Step 1: Break down the complex query into simpler subquestions
    try:
        subquestions = split_into_subquestions(query_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subquestion splitting failed: {e}")

    results = []

    # Step 2: Process each subquestion individually
    for subq in subquestions:
        try:
            # Classify the subquestion to determine if it's qualitative or quantitative
            query_type = classify_subquestion(subq)
        except Exception as e:
            results.append({"question": subq, "error": f"Classification failed: {e}"})
            continue

        # Step 3: Route the subquestion to the appropriate handler based on classification
        if query_type == "qualitative":
            # Handle narrative-based queries
            results.append(await handle_qualitative(subq))
        elif query_type == "quantitative":
            # Handle numerical queries
            results.append(await handle_quantitative(subq))
        else:
            # Handle unsupported query types 
            results.append({"question": subq, "type": query_type, "note": "Unsupported type"})

    # Step 4: Synthesize the results into a coherent summary
    try:
        summary = merge_responses(results)
    except Exception as e:
        summary = f"Summary generation failed: {e}"

    # Return the complete response with original query, individual results, and summary
    return {
        "original_query": query_text,  # The original complex query
        "subquestions": results,       # Individual results for each subquestion
        "summary": summary            # Synthesized summary of all results
    }
