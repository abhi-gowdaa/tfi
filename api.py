from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import aiohttp
from core.scraper import AsyncTransFiScraper
from core.embedder import AsyncEmbedder
from core.retrival import AsyncRetriever
import time
import json

from query import AsyncQueryEngine

app = FastAPI(title="TransFi RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vector_store = None
embeddings = None

@app.on_event("startup")
async def startup_event():
    global vector_store, embeddings
    from query import load_faiss_index

    print("Loading FAISS index and embeddings...")
    vector_store, embeddings = load_faiss_index()
    print(" FAISS index loaded successfully!")


class IngestRequest(BaseModel):
    urls: List[str]
    callback_url: str


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5


class BatchQueryRequest(BaseModel):
    questions: List[str]
    k: Optional[int] = 5
    callback_url: Optional[str] = None


async def send_webhook(callback_url: str, data: dict):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(callback_url, json=data, timeout=10) as response:
                print(f"  Webhook sent to {callback_url}: {response.status}")
    except Exception as e:
        print(f"  Failed to send webhook to {callback_url}: {str(e)}")


async def run_ingestion(urls: List[str], callback_url: str):
    start_time = time.time()
    
    try:
        all_documents = []
        scraper_metrics_all = []
        
        # Scrape all URLs
        for url in urls:
            scraper = AsyncTransFiScraper()
            documents = await scraper.scrape_all(str(url))
            all_documents.extend(documents)
            scraper_metrics_all.append(scraper.get_metrics())
        
        # Embed and index
        # embedder = AsyncEmbedder()
        # await embedder.process_documents(all_documents)
        # embedder_metrics = embedder.get_metrics()
        
        total_time = time.time() - start_time
        
        # Aggregate metrics
        total_scraped = sum(int(m['pages_scraped']) for m in scraper_metrics_all)
        total_failed = sum(int(m['pages_failed']) for m in scraper_metrics_all)
        
        metrics = {
            "status": "completed",
            "total_time": f"{total_time:.1f}s",
            "pages_scraped": total_scraped,
            "pages_failed": total_failed,
            # "total_chunks_created": embedder_metrics['chunks_created'],
            # "total_tokens_processed": embedder_metrics['tokens_processed'],
            # "embedding_generation_time": embedder_metrics['embedding_time'],
            # "indexing_time": embedder_metrics['indexing_time'],
            "documents_processed": len(all_documents)
        }
        
        await send_webhook(callback_url, {"metrics": metrics})
        
    except Exception as e:
        error_metrics = {
            "status": "failed",
            "error": str(e),
            "total_time": f"{time.time() - start_time:.1f}s"
        }
        await send_webhook(callback_url, {"metrics": error_metrics})


@app.post("/api/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        run_ingestion,
        [str(url) for url in request.urls],
        str(request.callback_url)
    )
    
    return {"message": "Ingestion started"}


@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        engine = AsyncQueryEngine(vector_store,embeddings)
        result = await engine.query_single(request.question, k=request.k)
        
        return {
            "answer": result['answer'],
            "sources": result['sources'],
            "metrics": result['metrics']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/batch")
async def query_batch(request: BatchQueryRequest, background_tasks: BackgroundTasks):
    
    
        # Synchronous response
    engine = AsyncQueryEngine(vector_store,embeddings)
    results = await engine.query_batch(request.questions, concurrent=True)
    
    aggregate_metrics = {
        "total_questions": len(results),
        "total_latency": f"{sum(r['metrics']['retrieval_time'] + r['metrics']['llm_time'] for r in results):.2f}s",
        "total_input_tokens": sum(r['metrics']['input_tokens'] for r in results),
        "total_output_tokens": sum(r['metrics']['output_tokens'] for r in results),
        "total_cost": f"${sum(r['metrics']['estimated_cost'] for r in results):.6f}"
    }
    
    return {
        "results": results,
        "aggregate_metrics": aggregate_metrics
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "TransFi RAG API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)