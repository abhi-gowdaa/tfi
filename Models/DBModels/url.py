from datetime import datetime
from Service.DBService.database import Base
from sqlalchemy import Column, Integer, String, Text,TIMESTAMP
from sqlalchemy.sql import func

class UrlMetadata(Base):
    __tablename__ = "url"
    
    id =Column(String, primary_key=True, index=True)
    url = Column(String,unique=True, index=True)
    status = Column(String, default="pending")
    created_at = Column(TIMESTAMP(timezone=True), default=func.now())
    completed_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())
    errors = Column(Text, nullable=True)
    
    
    