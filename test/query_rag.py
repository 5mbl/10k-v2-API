import os
import re
import uuid
from pinecone import Pinecone
from pinecone import ServerlessSpec  # ✅ Correct for Pinecone v5

from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
#from llama_index.readers.file import SimpleDirectoryReader

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.llms import llm  # ✅ Correct for LlamaIndex v0.10+
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# 🔗 Pinecone API Key & Setup
PINECONE_API_KEY = "pcsk_43C2Re_TS2J9enC1v2zFzUc3zkosWke62ZBL9UeSXhgL4UsGoHE3wJz1ohgc1B725G5y9S"  # 🔥 Replace with your API key
PINECONE_ENV = "us-west1-gcp"  # Check your Pinecone dashboard for the correct region
INDEX_NAME = "sec-filings"  # Name of Pinecone index

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)


if not INDEX_NAME:
    raise ValueError("INDEX_NAME is not defined or is None")

# ✅ Debugging Step
print(f"Using Pinecone Index: {INDEX_NAME}")

# ✅ Create Pinecone index if not exists
if INDEX_NAME not in [index_info['name'] for index_info in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")
else:
    print(f"✅ Pinecone index already exists: {INDEX_NAME}")

# ✅ Connect to Pinecone
pinecone_index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")

# ✅ Check if embeddings exist in Pinecone
stats = pinecone_index.describe_index_stats()
total_vectors = stats.get("total_vector_count", 0)
print("\n✅ Pinecone Index Stats:", stats)
if total_vectors > 0:
    print(f"✅ Embeddings already exist in Pinecone! Total vectors: {total_vectors}")
else:
    print("🔄 No embeddings found. Processing documents...")
    
    # 📂 Step 1: Load 10-K Filings from the Folder
    FOLDER_PATH = "./sec_filings"
    documents = SimpleDirectoryReader(input_dir=FOLDER_PATH).load_data()
    print(f"Loaded {len(documents)} documents from {FOLDER_PATH}")

    # ✂️ Step 2: Chunk the Documents
    node_parser = SentenceSplitter.from_defaults(
        chunk_size=512,
        chunk_overlap=128
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Chunked documents into {len(nodes)} nodes")

    # ✅ Manually Upsert Embeddings into Pinecone
    print("\n🔄 Generating embeddings and storing in Pinecone...")
    BATCH_SIZE = 50  # Adjust this if needed
    entries = []
    for node in nodes:
        vector = embed_model.get_text_embedding(node.text)
        entries.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": node.metadata
        })

    # ✅ Split entries into smaller batches
    def batch(iterable, n=BATCH_SIZE):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    # ✅ Upsert in batches
    for batch_entries in batch(entries, BATCH_SIZE):
        pinecone_index.upsert(batch_entries)
        print(f"✅ Upserted batch of {len(batch_entries)} vectors.")  

    print("✅ Successfully stored embeddings in Pinecone!")

# ✅ Create LlamaIndex Pinecone store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="sec_filings")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
print("Created query engine with top_k=5")

response = query_engine.query("What are Apple's risk factors in 2024?")
print("Queried Pinecone with LlamaIndex")
print("\n📌 AI Response:")
print(response)