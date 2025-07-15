import os
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Create router instead of app
router = APIRouter()

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("INDEX_NAME", "sec-filings")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("Missing OpenAI or Pinecone API keys or Index Name")

""" # Debug - Check if env variables are loaded
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}") # Mask for security
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENV: {PINECONE_ENV}")
print(f"INDEX_NAME: {INDEX_NAME}") """

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if not exists
existing_indexes = pc.list_indexes()
if INDEX_NAME not in [index['name'] for index in existing_indexes]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")
else:
    print(f"✅ Pinecone index already exists: {INDEX_NAME}")


# Connect to Pinecone
pinecone_index = pc.Index(INDEX_NAME)
print(f"✅ Connected to Pinecone index: {INDEX_NAME}")



# Check if embeddings exist in Pinecone
stats = pinecone_index.describe_index_stats()
total_vectors = stats.get("total_vector_count", 0)
print("\n✅ Pinecone Index Stats:", stats)

if total_vectors > 0:
    print(f"✅ Embeddings already exist in Pinecone! Total vectors: {total_vectors}")
else:
    print("🔄 No embeddings found. ")


# Create LlamaIndex Pinecone store
#vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="")
index = VectorStoreIndex.from_vector_store(vector_store=PineconeVectorStore(pinecone_index=pinecone_index))

""" # ✅ Create retriever from the index
retriever = index.as_retriever(similarity_top_k=5) """

# Create Query Engine
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

# Initialize the OpenAI Embedding Model
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)


# Pydantic model for query INPUT
class QueryRequest(BaseModel):
    query: str

# Pydantic model for API OUTPUT
class QueryResponse(BaseModel):
    query: str
    retrieved_texts: list[str]
    response: str




# ✅ FastAPI Endpoint for Query
@router.post("/rag", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    query_text = request.query

    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # ✅ Generate embeddings for query
        query_vector = embed_model.get_text_embedding(query_text)

        # ✅ Search Pinecone for top 5 matches
        search_results = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)
        
        # Extract text from matches
        retrieved_texts = []

        if search_results and "matches" in search_results:
            for match in search_results["matches"]:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "⚠️ No text found in metadata")
                retrieved_texts.append(text)
                print(f"🟢 Score: {match['score']}, Metadata: {metadata}")

        if not retrieved_texts:
            return {
                "query": query_text,
                "retrieved_texts": [],
                "response": "❌ No valid text found for query."
            }

        if retrieved_texts:
            # ✅ Use LLM to generate a response based on retrieved texts
            response = query_engine.query(query_text)
            result = response.response.strip() if response.response else "❌ No valid response from LLM"

        return {
            "query": query_text,
            "retrieved_texts": retrieved_texts,
            "response": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")













# Test Query
TEST_QUERY = "What are Apple's risk factors in 2024?"

# Generate embedding for query 
query_text = "What are the risk factors of apple?"
query_vector = embed_model.get_text_embedding(query_text)


# Search Pinecone for the top 5 similar vectors
search_results = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)


# Extract text from matches
retrieved_texts = []
print("\n🔍 Pinecone Search Results:")
if search_results and "matches" in search_results:
    for match in search_results["matches"]:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "⚠️ No text found in metadata")
        retrieved_texts.append(text)
        print(f"🟢 Score: {match['score']}, Metadata: {metadata}")
else:
    print("❌ No matches found in Pinecone! Check if data is indexed correctly.")

#  Generate a response using GPT-4/ gpt-3.5-turbo / gpt-4o-mini
if retrieved_texts:
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    index = VectorStoreIndex.from_vector_store(vector_store=PineconeVectorStore(pinecone_index=pinecone_index))
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

    response = query_engine.query(query_text)
    print("\n📌 AI Response:")
    print(response)
else:
    print("❌ No valid text found for query.")


