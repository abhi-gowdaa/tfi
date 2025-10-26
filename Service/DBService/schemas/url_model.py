from pydantic import BaseModel
from typing import Optional


class UrlModel(BaseModel):
    url_id: Optional[str] = None
    url:Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    errors: Optional[str] = None
    
    class Config:
        from_attributes = True
    
