import os
from typing import Optional

from pydantic import BaseModel
from Service.DBService.database import Database
from sqlalchemy.orm import Session
from Models.DBModels.url import UrlMetadata as UrlMetadataModel
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.environ['DATABASE_URL']

 
class create_url(BaseModel):
    url:str
    url_id:str

class UrlService:
    def __init__(self):
        self.DATABASE_URL=DATABASE_URL
        self.db_object=Database(self.DATABASE_URL)
    
    
    def add_url(self, url:str,url_id:str):
        db:Session=self.db_object.create_session()
        try:
            existing_url = db.query(UrlMetadataModel).filter(UrlMetadataModel.url == url).first()
            if existing_url:
                return existing_url
            
            url_metadata = UrlMetadataModel(id=url_id,url=url)
            db.add(url_metadata)
            db.commit()
            db.refresh(url_metadata)
            return url_metadata
        finally:
            self.db_object.close_session(db)
            
    def get_url_exist(self, url: str):
        db: Session = self.db_object.create_session()
        try:
            existing_url = db.query(UrlMetadataModel).filter(UrlMetadataModel.url == url).first()
            return existing_url
        finally:
            self.db_object.close_session(db)
        
        
        
    def update_status(self,url_id: str,status: str,error: Optional[str] = None ):
        db: Session = self.db_object.create_session()
        try:
            url_record = db.query(UrlMetadataModel).filter( UrlMetadataModel.id == url_id).first()
            
            if not url_record:
                raise ValueError(f"URL with ID {url_id} not found")
            
          
            url_record.status = status
            
            
            if error:
                url_record.errors = error
                print(error)
            else:
                if status != "failed":
                    url_record.errors = None
            
            db.commit()
            db.refresh(url_record)

            return url_record
            
        except Exception as e:
            db.rollback()
            raise
        finally:
            self.db_object.close_session(db)
            
    def get_url_by_id(self, url_id: str):
        db: Session = self.db_object.create_session()
        try:
            url_record = db.query(UrlMetadataModel).filter(UrlMetadataModel.id == url_id).first()
            return url_record
        finally:
            self.db_object.close_session(db)

    def get_all_urls(self):
        db: Session = self.db_object.create_session()
        try:
            all_urls = db.query(UrlMetadataModel).order_by(UrlMetadataModel.created_at.desc()).all()
            return all_urls
        finally:
            self.db_object.close_session(db)
    

