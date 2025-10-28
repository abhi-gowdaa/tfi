import asyncio
import time
from typing import List, Dict, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import tiktoken
from pathlib import Path


class AsyncRetriever:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.vector_store = None
        
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index"""
        try:
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"  Loaded FAISS index from: {self.index_path}")
        except Exception as e:
            raise ValueError(f"Failed to load FAISS index from {self.index_path}: {str(e)}")
    
    async def retrieve_documents(self, query: str, k: int = 5) -> Tuple[List, float]:
        """
        Retrieve relevant documents for a query
        
        Returns:
            Tuple of (documents with scores, retrieval_time)
        """
        start_time = time.time()
        
        # Run similarity search in executor to not block
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            self.vector_store.similarity_search_with_score,
            query,
            k
        )
        
        retrieval_time = time.time() - start_time
        
        return results, retrieval_time
    
    def create_prompt(self, context: str, question: str) -> str:
            """Create a prompt for the LLM"""
            return f"""You are an expert assistant for TransFi, a global payment platform.

    Context from TransFi documentation:
    {context}

    Question: {question}

    Instructions:
    1. Answer based ONLY on the provided context above
    2. Be concise and clear (50-150 words)
    3. If the answer is not in the context, say "This information is not available in the provided context"
    4. Cite specific features, products, or solutions when mentioning them
    5. If greeting, respond warmly and offer to help with TransFi questions

    Answer:"""
        
    async def generate_answer(self, prompt: str) -> Tuple[str, float, int, int]:
        """
        Generate answer using LLM
        
        Returns:
            Tuple of (answer, llm_time, input_tokens, output_tokens)
        """
        start_time = time.time()
        
        # Count input tokens
        input_tokens = len(self.encoding.encode(prompt))
        
        # Generate response in executor
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            self.llm.invoke,
            prompt
        )
        
        llm_time = time.time() - start_time
        
        # Count output tokens
        output_tokens = len(self.encoding.encode(response.content))
        
        return response.content, llm_time, input_tokens, output_tokens
    
    async def query(self, question: str, k: int = 5) -> Dict:
        """
        Complete query pipeline: retrieve -> generate -> format
        
        Returns:
            Dict with answer, sources, and metrics
        """
        total_start = time.time()
        
        # Step 1: Retrieve documents
        results, retrieval_time = await self.retrieve_documents(question, k)
        
        # Step 2: Prepare context and sources
        context_parts = []
        sources = []
        
        for doc, score in results[:3]:  # Use top 3 for context
            context_parts.append(doc.page_content)
            sources.append({
                'url': doc.metadata.get('url', 'N/A'),
                'title': doc.metadata.get('title', 'N/A'),
                'category': doc.metadata.get('category', 'N/A'),
                'snippet': doc.page_content[:200] + "...",
                'relevance_score': float(score)
            })
        
        context = "\n\n".join(context_parts)
        prompt = self.create_prompt(context, question)
        
        # Step 3: Generate answer
        answer, llm_time, input_tokens, output_tokens = await self.generate_answer(prompt)
        
        # Step 4: Calculate metrics
        total_latency = time.time() - total_start
        post_processing_time = total_latency - retrieval_time - llm_time
        
        # Estimate cost (Gemini pricing - approximate)
        estimated_cost = (input_tokens * 0.00001 + output_tokens * 0.00003) / 1000
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'metrics': {
                'total_latency': total_latency,
                'retrieval_time': retrieval_time,
                'llm_time': llm_time,
                'post_processing_time': post_processing_time,
                'documents_retrieved': len(results),
                'documents_used_in_answer': len(sources),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'estimated_cost': estimated_cost
            }
        }
    
    async def query_batch(self, questions: List[str], k: int = 5, concurrent: bool = False) -> List[Dict]:
        """
        Process multiple questions
        
        Args:
            questions: List of questions
            k: Number of documents to retrieve per question
            concurrent: If True, process all questions concurrently
        
        Returns:
            List of results
        """
        if concurrent:
            # Process all questions concurrently
            tasks = [self.query(q, k) for q in questions]
            results = await asyncio.gather(*tasks)
            return list(results)
        else:
            # Process sequentially
            results = []
            for question in questions:
                result = await self.query(question, k)
                results.append(result)
            return results