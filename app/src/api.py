from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from crawler import crawl_urls, get_new_urls
from db import insert_passages_to_chromadb
from generate_response import evaluate_marketing_copy, generate_response

app = FastAPI(
    title="PharmaCopy Compliance API",
    description="A FastAPI service for crawling FDA guideline URLs, indexing passages into ChromaDB, and evaluating pharmaceutical marketing copy.",
    version="0.1.0",
)

DEFAULT_OUTPUT_PATH = Path("data/output.jsonl")
DEFAULT_SEEN_PATH = Path("data/seen_urls.txt")
DEFAULT_URLS_FILE = Path("data/urls.txt")
DEFAULT_COLLECTION = "pharma_copy_collection"


class CrawlPayload(BaseModel):
    urls: Optional[List[str]] = None
    urls_file: Optional[Path] = DEFAULT_URLS_FILE
    output_path: Path = DEFAULT_OUTPUT_PATH
    seen_path: Path = DEFAULT_SEEN_PATH


class IndexPayload(BaseModel):
    output_path: Path = DEFAULT_OUTPUT_PATH
    collection_name: str = DEFAULT_COLLECTION


class EvaluatePayload(BaseModel):
    marketing_copy: str
    collection_name: str = DEFAULT_COLLECTION


class AskPayload(BaseModel):
    question: str
    collection_name: str = DEFAULT_COLLECTION


@app.on_event("startup")
async def startup_event():
    """Pre-load the LLM model on startup to avoid blocking first requests."""
    try:
        print("Pre-loading LLM model...")
        # This will trigger model loading
        from generate_response import ResponseGenerator
        generator = ResponseGenerator.get_instance()
        generator._load_model()
        print("Model pre-loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to pre-load model: {e}")
        print("Model will be loaded on first request instead.")


@app.get("/")
def root():
    return {
        "service": "PharmaCopy Compliance API",
        "status": "ok",
        "endpoints": [
            "/health",
            "/crawl",
            "/index",
            "/evaluate",
            "/ask",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/crawl")
def crawl(payload: CrawlPayload):
    if payload.urls:
        urls = payload.urls
    else:
        urls_file = Path(payload.urls_file)
        if not urls_file.exists():
            raise HTTPException(status_code=400, detail=f"URLs file not found: {urls_file}")
        urls = get_new_urls(str(urls_file), str(payload.seen_path))

    if not urls:
        return {
            "crawled": 0,
            "message": "No new URLs found to crawl.",
        }

    crawled_count = crawl_urls(urls, str(payload.output_path), str(payload.seen_path))
    return {
        "crawled": crawled_count,
        "output_path": str(payload.output_path),
        "seen_path": str(payload.seen_path),
    }


@app.post("/index")
def index(payload: IndexPayload):
    output_path = Path(payload.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=400, detail=f"Output file not found: {output_path}")

    try:
        insert_passages_to_chromadb(str(output_path), collection_name=payload.collection_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "collection_name": payload.collection_name,
        "output_path": str(output_path),
    }


@app.post("/evaluate")
def evaluate(payload: EvaluatePayload):
    if not payload.marketing_copy.strip():
        raise HTTPException(status_code=400, detail="marketing_copy must not be empty")

    try:
        evaluation = evaluate_marketing_copy(payload.marketing_copy, collection_name=payload.collection_name)
        return {
            "evaluation": evaluation,
            "collection_name": payload.collection_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(payload: AskPayload):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        answer = generate_response(payload.question, collection_name=payload.collection_name)
        return {
            "answer": answer,
            "collection_name": payload.collection_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
