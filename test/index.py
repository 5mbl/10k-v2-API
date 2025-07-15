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
# Embeddings Model OpenAI
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)


# Helper Functions

def is_file_already_indexed(file_name):
    """Check if a file is already indexed in Pinecone."""
    query_results = pinecone_index.query(
        vector=[0] * 1536,  # Dummy vector (just to use metadata filtering)
        filter={"file_name": file_name},  # ✅ Check based on file name
        top_k=1,  # Only need one match to confirm existence
        include_metadata=True
    )
    return len(query_results["matches"]) > 0  # True if file exists


def extract_metadata(file_path: str):
    """Extracts company name and year from a filename like 'apple_2024.pdf'."""
    file_name = Path(file_path).stem  # Get filename without extension
    match = re.match(r"([a-zA-Z]+)_(\d{4})", file_name)  # Match 'company_year'
    if match:
        company_name, year = match.groups()
        return {"company": company_name.capitalize(), "year": int(year)}
    return {"company": "Unknown", "year": "Unknown"}


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


# 📂 Step 1: Load 10-K Filings from the Folder
FOLDER_PATH = "./sec_filings"
documents = SimpleDirectoryReader(input_dir=FOLDER_PATH).load_data()
print(f"Loaded {len(documents)} documents from {FOLDER_PATH}")


# Avoid re-indexing if file is already indexed
for doc in documents:
    file_path = doc.metadata.get("file_path", "UNKNOWN_PATH")
    file_name = Path(file_path).name  # Extract just the filename

    # ✅ Check if file is already indexed
    if is_file_already_indexed(file_name):
        print(f"🚫 Skipping already indexed file: {file_name}")
        continue  # Skip indexing if file is found in Pinecone

    print(f"✅ Indexing new file: {file_name}")


# ✂️ Step 2: Chunk the Documents
node_parser = SentenceSplitter.from_defaults(
    chunk_size=512,
    chunk_overlap=128
)

nodes = node_parser.get_nodes_from_documents(documents)
print(f"Chunked documents into {len(nodes)} nodes")

# print("TEXT",nodes[1].text) # check text chunk of the node
# print("METADATA",nodes[1].metadata) # check metadata

# Manually Upsert Embeddings into Pinecone
print("\n🔄 Generating embeddings and storing in Pinecone...")
BATCH_SIZE = 50  # Adjust this if needed
entries = []
for node in nodes:
    metadata = extract_metadata(node.metadata.get("file_path", "Unknown"))

    ## ✅ Ensure text is not empty before embedding
    if not node.text.strip():
        print(f"⚠️ Skipping empty chunk for {node.metadata.get('file_name', 'Unknown')}")
        continue  # Skip empty chunks

    vector = embed_model.get_text_embedding(node.text)
    entries.append({
        "id": str(uuid.uuid4()),
        "values": vector,
        "metadata": {
            "text": node.text,  # ✅ Store actual text
            "company": metadata["company"],
            "year": metadata["year"],
            "file_name": node.metadata.get("file_name", "Unknown"),
            "page_label": node.metadata.get("page_label", "Unknown"),
            "file_path": node.metadata.get("file_path", "Unknown"),
        }
    })

# ✅ Split entries into smaller batches
def batch(iterable, n=BATCH_SIZE):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# ✅ Upsert in batches
for batch_entries in batch(entries, BATCH_SIZE):
    pinecone_index.upsert(batch_entries)
    print(f"✅ Upserted batch of {len(batch_entries)} vectors.")  

print("✅ Successfully stored embeddings in Pinecone!")



