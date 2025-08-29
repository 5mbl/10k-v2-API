import asyncio
from qualitative import handle_qualitative
from quantitative import handle_quantitative
from utils import split_into_subquestions, classify_subquestion, merge_responses

# Remove the .routes from the other files in this dir, in order to make the tests

async def test_qualitative():
    query = "What are Apple's biggest risks in 2024?"
    result = await handle_qualitative(query)
    print("Qualitative Test Result:")
    print(result)

async def test_quantitative():
    query = "What was Tesla's net income in 2022?"
    result = await handle_quantitative(query)
    print(" Quantitative Test Result:")
    print(result)

def test_utils():
    query = "What are Apple's risks and Tesla's revenue in 2022?"
    subqs = split_into_subquestions(query)
    print("Subquestions:", subqs)

    for q in subqs:
        label = classify_subquestion(q)
        print(f"Classification for '{q}':", label)

    sample_results = [
        {"type": "qualitative", "question": "What are Apple's risks?", "response": "Apple faces supply chain and market risks."},
        {"type": "quantitative", "question": "What is Tesla's revenue in 2022?", "value": "$81.5B"}
    ]
    summary = merge_responses(sample_results)
    print("Summary:", summary)

if __name__ == "__main__":
    print("Running Independent Route Tests\n")

    # Test utils (no await needed)
    # test_utils()

    # Run async tests
    # asyncio.run(test_qualitative())
    # asyncio.run(test_quantitative())
