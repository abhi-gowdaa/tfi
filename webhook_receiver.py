from fastapi import FastAPI, Request
import uvicorn
import argparse
from datetime import datetime
import json

app = FastAPI(title="Webhook Receiver")


@app.post("/webhook")
async def receive_webhook(request: Request):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = await request.json()
    
    print("\n" + "=" * 70)
    print(f" WEBHOOK RECEIVED at {timestamp}")
    print("=" * 70)
    
    if "metrics" in data:
        metrics = data["metrics"]
        print("\n Ingestion Metrics:")
        print(f"  Status: {metrics.get('status', 'N/A')}")
        print(f"  Total Time: {metrics.get('total_time', 'N/A')}")
        print(f"  Pages Scraped: {metrics.get('pages_scraped', 'N/A')}")
        print(f"  Pages Failed: {metrics.get('pages_failed', 'N/A')}")
        print(f"  Chunks Created: {metrics.get('total_chunks_created', 'N/A')}")
        print(f"  Tokens Processed: {metrics.get('total_tokens_processed', 'N/A')}")
        print(f"  Embedding Time: {metrics.get('embedding_generation_time', 'N/A')}")
        print(f"  Indexing Time: {metrics.get('indexing_time', 'N/A')}")
        
        if metrics.get('status') == 'failed':
            print(f"\n  Error: {metrics.get('error', 'Unknown error')}")
    
    elif "results" in data:
        results = data["results"]
        aggregate = data.get("aggregate_metrics", {})
        
        print("\n Batch Query Results:")
        print(f"  Total Questions: {aggregate.get('total_questions', len(results))}")
        print(f"  Total Latency: {aggregate.get('total_latency', 'N/A')}")
        print(f"  Total Input Tokens: {aggregate.get('total_input_tokens', 'N/A')}")
        print(f"  Total Output Tokens: {aggregate.get('total_output_tokens', 'N/A')}")
        print(f"  Total Cost: ${aggregate.get('total_cost', 'N/A')}")
        
        print("\n Individual Results:")
        for i, result in enumerate(results, 1):
            print(f"\n  Question {i}: {result['question']}")
            print(f"  Answer: {result['answer'][:100]}...")
    
    else:
        print("\n Raw Data:")
        print(json.dumps(data, indent=2))
    
    print("\n" + "=" * 70)
    
    return {"status": "received", "timestamp": timestamp}


@app.get("/")
async def root():
    return {"message": "Webhook receiver is running"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webhook receiver for TransFi RAG')
    parser.add_argument('--port', type=int, default=8001, help='Port to run on')
    
    args = parser.parse_args()
    
    print(f"🎧 Starting webhook receiver on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)