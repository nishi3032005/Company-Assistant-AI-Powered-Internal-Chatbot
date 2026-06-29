"""
main.py
--------
FastAPI backend for the Company Assistant chatbot.

Endpoints:
  GET  /            → health-check / welcome message
  GET  /health      → JSON health status
  POST /chat        → accepts a query, returns answer + sources
  GET  /documents   → lists all indexed document sources
"""
from dotenv import load_dotenv
load_dotenv()

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


# Pydantic schemas


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000,
                       description="The employee's question")
    top_k: int = Field(default=4, ge=1, le=10,
                       description="Number of document chunks to retrieve")


class SourceDocument(BaseModel):
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str



# App lifecycle — build the RAG index once at startup


rag: RAGPipeline | None = None   # global reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the FAISS index when the server starts."""
    global rag
    print(" Starting Company Assistant API …")
    rag = RAGPipeline(top_k=4)
    print(" RAG pipeline ready. Server is live.")
    yield
    # cleanup (if needed) goes here
    print(" Shutting down.")



# FastAPI application


app = FastAPI(
    title="Company Assistant API",
    description="AI-powered internal chatbot using RAG (FAISS + Claude)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # restrict to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)



# Routes


@app.get("/", summary="Welcome")
def root() -> dict[str, str]:
    return {
        "message": "Company Assistant API is running 🎉",
        "docs": "Visit /docs for the interactive API documentation.",
    }


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> Any:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialised yet.")
    return {
        "status": "ok",
        "index_size": rag.index.ntotal if rag.index else 0,
        "model": rag.model_name,
    }


@app.post("/chat", response_model=ChatResponse, summary="Ask the company assistant")
def chat(request: ChatRequest) -> Any:
    """
    Main endpoint.

    - Embeds the user query
    - Retrieves the top-k relevant document chunks from FAISS
    - Calls Claude to generate a context-grounded answer
    - Returns the answer + source passages
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    # Temporarily override top_k if the caller specified one
    original_top_k = rag.top_k
    rag.top_k = request.top_k

    t0 = time.perf_counter()
    try:
        result = rag.query(request.query)
    except Exception as exc:
        # Re-raise as HTTP 500 with the error message
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        rag.top_k = original_top_k   # restore

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": request.query,
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_ms": round(latency_ms, 2),
    }


@app.get("/documents", summary="List indexed document sources")
def list_documents() -> Any:
    """Returns the unique document sources that have been indexed."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    sources = sorted({p["source"] for p in rag.passages})
    return {
        "total_passages": len(rag.passages),
        "sources": sources,
    }
