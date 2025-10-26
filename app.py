from typing import List, Optional
from Models.pymodels import IngestResponse, QueryRequest, create_url
from Service.DBService.url_service import UrlService
from Service.embedding import get_prompt, search_similar
from fastapi import FastAPI,HTTPException
from redis import Redis
from rq import Queue
from pydantic import BaseModel
import json
import uvicorn
from worker import process_ingest_url
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_conn=Redis(host="localhost",port=6379)

task_queue=Queue("task_queue",connection=redis_conn)


    
@app.get("/")
def ping():
    return {"message":"server running"}


@app.post("/ingest-url")
async def ingest_url(request: create_url):
    try:
        url=str(request.url)
        url_id=str(uuid4())
        
        url_service=UrlService()
        existing_job = url_service.get_url_exist(url)
        
        if existing_job:
            return IngestResponse(
                job_id=str(existing_job.id),
                status=existing_job.status,
                message=f"URL already exists in db with status: {existing_job.status}"
            )
       
        job_detail=url_service.add_url(url=url,url_id=url_id)
        print(job_detail)
        
        #redis queue worker 
      
        job_instance=process_ingest_url(url,url_id)
            
        
        print(job_instance)
        
        return IngestResponse(
            job_id=url_id,
            status="pending",
            message="URL queued for processing"
        )
        
    except Exception as e:
        raise HTTPException(f"Error in adding url : {e}")
        

 
@app.post("/query")
async def query_embeddings(request: QueryRequest):
    try:
        results = search_similar(
            query=request.query,
            k=request.k,
            url_id=request.url_id
        )
        
        
        text_results = results['documents'][0]
        
        # print(text_results)
        prompt = get_prompt(text_results, request.query)
        result = llm.invoke(prompt)
        
        return result.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying: {e}")



@app.get("/status/{url_id}")
async def get_status(url_id: str):
    try:
        url_service = UrlService()
        url_data = url_service.get_url_by_id(url_id)
        
        if not url_data:
            raise HTTPException(status_code=404, detail="URL not found")
        
        return {
            "id": url_data.id,
            "url": url_data.url,
            "status": url_data.status,
            "error": url_data.errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-urls")
async def list_urls():
    try:
        url_service = UrlService()
        all_urls = url_service.get_all_urls()
        
        return [
            {
                "id": url.id,
                "url": url.url,
                "status": url.status
            }
            for url in all_urls
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__=="__main__":
    
    uvicorn.run(app,host="localhost",port=8000)