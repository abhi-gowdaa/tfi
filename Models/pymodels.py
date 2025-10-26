from typing import Optional
from pydantic import BaseModel


class create_url(BaseModel):
    url:str
    
class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str
    
class QueryRequest(BaseModel):
    query: str
    url_id: Optional[str] = None
    k: Optional[int] = 5

