import logging

LOG_FMT = "%(levelname)s %(name)s – %(message)s"

logging.basicConfig(
    level=logging.INFO,          # or DEBUG
    format=LOG_FMT,
    handlers=[logging.StreamHandler()]
)

from fastapi import FastAPI
from dotenv import load_dotenv
import os


load_dotenv()

if not os.getenv("OPENAI_API_KEY") or not os.getenv("POLYGON_API_KEY"):
    raise ValueError("API keys are missing from environment variables")

# from routers import classify, financial, rag 

from routes.hybrid_query import router as hybrid_router


from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from Next.js app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Register routers
""" app.include_router(classify.router, prefix="/api", tags=["classify"])
app.include_router(financial.router, prefix="/api", tags=["financial"])
app.include_router(rag.router, prefix="/api", tags=["rag"]) """

app.include_router(hybrid_router, prefix="/api", tags=["hybrid"])



# Root endpoint (optional)
@app.get("/")
def root():
    return {"message": "Welcome to the SEC-RAG-API"}
