import asyncio
import time
import tiktoken
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()


class AsyncEmbedder:
    """
    Async-first embedder for chunking, embedding generation, and FAISS indexing.
    Handles batch processing and provides comprehensive metrics.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        index_path: str = "faiss_index",
        batch_size: int = 5,  # Reduced from 10 to avoid rate limits
        delay_between_batches: float = 2.0  # 2 seconds between batches
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables.\n"
                "Please create a .env file with: GOOGLE_API_KEY=your_api_key_here\n"
                "Or set it in your environment: export GOOGLE_API_KEY=your_api_key"
            )
        
        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Metrics tracking
        self.metrics = {
            'chunks_created': 0,
            'tokens_processed': 0,
            'embedding_time': 0,
            'indexing_time': 0,
            'chunking_time': 0,
            'documents_processed': 0,
            'batches_processed': 0
        }
        
        self.vector_store = None
        
        print(f"  Embedder initialized with batch_size={batch_size}, delay={delay_between_batches}s")
    
    def chunk_document(self, document: Dict) -> List[Document]:
        """
        Chunk a single document into smaller pieces.
        
        Args:
            document: Dict with 'long_description', 'url', 'title', etc.
        
        Returns:
            List of LangChain Document objects with metadata
        """
        content = document.get('long_description', '')
        if not content:
            # Fallback to short description if long description is empty
            content = document.get('short_description', '')
        
        if not content:
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'url': document.get('url', ''),
                    'title': document.get('title', ''),
                    'category': document.get('category', ''),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'short_description': document.get('short_description', '')[:200]
                }
            )
            documents.append(doc)
        
        return documents
    
    async def chunk_documents_batch(self, documents: List[Dict]) -> List[Document]:
        """
        Chunk multiple documents concurrently.
        
        Args:
            documents: List of document dicts
        
        Returns:
            List of all chunked Document objects
        """
        start_time = time.time()
        
        print(f"📝 Chunking {len(documents)} documents...")
        
        # Run chunking in executor (CPU-bound operation)
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.chunk_document, doc)
            for doc in documents
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_chunks = []
        for chunks in results:
            all_chunks.extend(chunks)
        
        self.metrics['chunking_time'] = time.time() - start_time
        self.metrics['chunks_created'] = len(all_chunks)
        
        print(f"  Created {len(all_chunks)} chunks in {self.metrics['chunking_time']:.2f}s")
        
        return all_chunks
    
    def count_tokens(self, texts: List[str]) -> int:
        """Count total tokens in a list of texts."""
        total = 0
        for text in texts:
            total += len(self.encoding.encode(text))
        return total
    
    async def create_or_update_index(self, documents: List[Document]) -> None:
        """
        Create a new FAISS index incrementally in batches.
        
        Args:
            documents: List of Document objects
        """
        start_time = time.time()
        
        print(f"💾 Creating FAISS index with {len(documents)} documents...")
        
        # Create initial FAISS index with first batch
        first_batch_size = min(self.batch_size, len(documents))
        first_batch = documents[:first_batch_size]
        
        print(f"  Creating initial index with {first_batch_size} documents...")
        
        # Create initial index with retry logic
        max_retries = 3
        retry_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                self.vector_store = await asyncio.get_event_loop().run_in_executor(
                    None,
                    FAISS.from_documents,
                    first_batch,
                    self.embeddings
                )
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"      Rate limit hit during index creation, waiting {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
                else:
                    raise
        
        # Add remaining documents in batches
        if len(documents) > first_batch_size:
            remaining_docs = documents[first_batch_size:]
            batches_added = 0
            
            for i in range(0, len(remaining_docs), self.batch_size):
                batch = remaining_docs[i:i + self.batch_size]
                
                # Add batch with retry logic
                for attempt in range(max_retries):
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.vector_store.add_documents,
                            batch
                        )
                        break
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower():
                            if attempt < max_retries - 1:
                                print(f"      Rate limit hit, waiting {retry_delay}s...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                raise
                        else:
                            raise
                
                batches_added += 1
                print(f"  Added batch {batches_added}/{(len(remaining_docs) + self.batch_size - 1) // self.batch_size}")
                
                # Delay between batches
                if i + self.batch_size < len(remaining_docs):
                    await asyncio.sleep(self.delay_between_batches)
        
        # Save to disk
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.vector_store.save_local,
            str(self.index_path)
        )
        
        self.metrics['indexing_time'] = time.time() - start_time
        
        print(f"  FAISS index created and saved to {self.index_path}")
        print(f"   Indexing time: {self.metrics['indexing_time']:.2f}s")
    
    async def process_documents(self, documents: List[Dict]) -> None:
        """
        Complete pipeline: chunk -> index (with embeddings).
        
        Args:
            documents: List of scraped document dicts
        """
        self.metrics['documents_processed'] = len(documents)
        
        # Step 1: Chunk all documents
        chunked_docs = await self.chunk_documents_batch(documents)
        
        if not chunked_docs:
            print("    No chunks created. Nothing to embed.")
            return
        
        # Count tokens for metrics
        texts = [doc.page_content for doc in chunked_docs]
        tokens = await asyncio.get_event_loop().run_in_executor(
            None,
            self.count_tokens,
            texts
        )
        self.metrics['tokens_processed'] = tokens
        print(f" Total tokens to process: {tokens:,}")
        
        # Step 2: Create FAISS index (embeddings generated during this step)
        await self.create_or_update_index(chunked_docs)
        
        print(f"\n  Embedding pipeline complete!")
    
    def get_metrics(self) -> Dict:
        """Return all metrics."""
        # Calculate embedding time from indexing time since they happen together
        embedding_time = self.metrics['indexing_time']
        
        return {
            'documents_processed': self.metrics['documents_processed'],
            'chunks_created': self.metrics['chunks_created'],
            'tokens_processed': self.metrics['tokens_processed'],
            'chunking_time': f"{self.metrics['chunking_time']:.2f}s",
            'embedding_time': f"{embedding_time:.2f}s",
            'indexing_time': f"{self.metrics['indexing_time']:.2f}s",
            'avg_tokens_per_chunk': (
                self.metrics['tokens_processed'] // self.metrics['chunks_created']
                if self.metrics['chunks_created'] > 0 else 0
            )
        }