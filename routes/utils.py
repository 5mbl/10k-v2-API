import os
import json
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
import re



# Load environment variables from .env file

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))


# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("INDEX_NAME", "sec-filings")

# Shared LLM and Embedder
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

# Utils


def split_into_subquestions(query: str):
    prompt = f"""
    Split the following query into independent, answerable subquestions.
    ONLY include questions actually implied by the original query!
    Return the list in valid JSON format:
    ["subquestion1", "subquestion2"]

    Query: "{query}"
    """
    response = llm.complete(prompt).text.strip()

    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try extracting JSON array via regex
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Fallback JSON parsing failed: {e}\nResponse was:\n{response}")
        raise ValueError(f"Failed to parse subquestions: {response}")

def classify_subquestion(subq: str):
    prompt = f"""
    Classify the following query:
    - 'quantitative': for numeric financial data
    - 'qualitative': for strategic/operational content
    Respond with one word only.
    Query: "{subq}"
    """
    return llm.complete(prompt).text.strip().lower()

def merge_responses(results):
    prompt = f"""
    Here are multiple subquestion answers:
    {json.dumps(results, indent=2)}
    Generate a single coherent paragraph summarizing the overall answer.
    """
    return llm.complete(prompt).text.strip()
