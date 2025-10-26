import Models.DBModels
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pkgutil
import importlib
import os
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.environ['DATABASE_URL']


Base = declarative_base()



engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
class Database:
    def __init__(self, database_url: str):
        self._engine = create_engine(database_url)
        self._SessionLocal = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)
        for _, module_name, _ in pkgutil.iter_modules(Models.DBModels.__path__):
            importlib.import_module(f"Models.DBModels.{module_name}")
        
  
        Base.metadata.create_all(self._engine)

    def get_engine(self) -> Engine:
        return self._engine

    def get_session_local(self) -> sessionmaker:
        return self._SessionLocal

    def create_session(self) -> Session:
        """Creates and returns a new session"""
        return self._SessionLocal()

    def close_session(self, session: Session):
        """Closes the given session"""
        session.close()

    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(self._engine)