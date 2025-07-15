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

class HybridQuery(BaseModel):
    query: str

@router.post("/hybrid-query")
async def hybrid_query(input: HybridQuery):
    query_text = input.query

    try:
        subquestions = split_into_subquestions(query_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subquestion splitting failed: {e}")

    results = []

    for subq in subquestions:
        try:
            query_type = classify_subquestion(subq)
        except Exception as e:
            results.append({"question": subq, "error": f"Classification failed: {e}"})
            continue

        if query_type == "qualitative":
            results.append(await handle_qualitative(subq))
        elif query_type == "quantitative":
            results.append(await handle_quantitative(subq))
        else:
            results.append({"question": subq, "type": query_type, "note": "Unsupported type"})

    try:
        summary = merge_responses(results)
    except Exception as e:
        summary = f"Summary generation failed: {e}"

    return {
        "original_query": query_text,
        "subquestions": results,
        "summary": summary
    }
