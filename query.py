import asyncio
import argparse
import time
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import tiktoken

load_dotenv()

# Initialize components globally
INDEX_PATH = "faiss_index"

 
def load_faiss_index():
    """Load existing FAISS index"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    try:
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store, embeddings
    except Exception as e:
        print(f"Creating new FAISS Web index at: {INDEX_PATH}")
        vector_store = FAISS.from_texts(["init"], embedding=embeddings)
        return vector_store, embeddings



class AsyncQueryEngine:
    def __init__(self, vector_store, embeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings
        
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key
        )
        
        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def get_prompt(self, context: str, question: str) -> str:
        """Create prompt for LLM"""
        return f"""You are an expert assistant for TransFi, a global payment platform.

Analyze the user's question carefully and provide a clear, helpful answer using ONLY the information in the context below.

Context from TransFi documentation:
{context}

Question: {question}

Instructions:
1. Identify the main intent of the question.
2. Check if the answer exists in the context.
3. Highlight the parts of the context that support your answer.
4. Provide a concise, clear answer (50-150 words).
5. If the answer is not in the context, say: "This information is not available in the provided context."
6. If it's a greeting, respond warmly and offer to help with TransFi questions.
7. If only partial information is available, give a short, relevant answer with reasoning.
8. Cite specific features, products, or solutions when mentioning them.

Answer:"""
    
    async def query_single(self, question: str, k: int = 5) -> Dict:
        """
        Process a single question through the RAG pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
        
        Returns:
            Dict with answer, sources, and metrics
        """
        total_start = time.time()
        
        # Step 1: Retrieval
        retrieval_start = time.time()
        
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            self.vector_store.similarity_search_with_score,
            question,
            k
        )
        
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Prepare context and sources
        context_parts = []
        sources = []
        seen_urls = set()  # Track unique sources
        
        # Use top 3 most relevant and unique documents
        for doc, score in results:
            url = doc.metadata.get('url', 'N/A')
            
            # Skip if we already have this URL  
            if url in seen_urls:
                continue
            
            seen_urls.add(url)
            context_parts.append(doc.page_content)
            
            # Get clean content snippet - first sentence or first 100 chars
            content = doc.page_content.strip()
            
            # Try to get first sentence
            sentences = content.split('.')
            if sentences and len(sentences[0]) < 150:
                snippet = sentences[0].strip() + "..."
            else:
                # Fallback to first 100 chars
                snippet = content[:100].strip() + "..."
            
            # Clean up the snippet
            snippet = snippet.replace('\n', ' ').replace('  ', ' ')
            
            # Get title - clean it up if too long
            full_title = doc.metadata.get('title', 'TransFi')
            
            # Simplify title if it's too long or has extra text
            if len(full_title) > 50 or ' - ' in full_title:
                # Extract the main product/solution name
                title_parts = full_title.split(' - ')
                title = title_parts[0].strip()
            else:
                title = full_title
            
            # Further simplification based on category
            category = doc.metadata.get('category', '')
            if category == 'product':
                # Extract product name (e.g., "BizPay" from longer title)
                words = title.split()
                if len(words) > 3:
                    # Take first 2-3 words
                    title = ' '.join(words[:2])
            
            sources.append({
                'title': title,
                'url': url,
                'snippet': snippet
            })
            
            # Stop after getting 2-3 unique sources
            if len(sources) >= 3:
                break
        
        context = "\n\n".join(context_parts)
        prompt = self.get_prompt(context, question)
        
        # Count input tokens
        input_tokens = len(self.encoding.encode(prompt))
        
        # Step 3: LLM Generation
        llm_start = time.time()
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            self.llm.invoke,
            prompt
        )
        
        llm_time = time.time() - llm_start
        
        # Count output tokens
        output_tokens = len(self.encoding.encode(response.content))
        
        # Step 4: Calculate metrics
        total_latency = time.time() - total_start
        post_processing_time = total_latency - retrieval_time - llm_time
        
        # Estimate cost (Gemini pricing - approximate)
        estimated_cost = (input_tokens * 0.00001 + output_tokens * 0.00003) / 1000
        
        return {
            'question': question,
            'answer': response.content,
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
    
    async def query_batch(self, questions: List[str], concurrent: bool = False) -> List[Dict]:
        """
        Process multiple questions.
        
        Args:
            questions: List of questions
            concurrent: If True, process all questions concurrently
        
        Returns:
            List of results
        """
        if concurrent:
            tasks = [self.query_single(q) for q in questions]
            results = await asyncio.gather(*tasks)
            return list(results)
        else:
            results = []
            for question in questions:
                result = await self.query_single(question)
                results.append(result)
            return results


def print_result(result: Dict, show_header: bool = True):
    """Print a single query result in the exact required format"""
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    
    print("Sources:")
    for i, source in enumerate(result['sources'], 1):
        source_name = source['title']
        # Format as "Title - URL" or just "Title" if URL looks cleaner
        if source['url'] != 'N/A':
            print(f"  {i}. {source_name} - {source['url']}")
        else:
            print(f"  {i}. {source_name}")
        print(f"     Snippet: \"{source['snippet']}\"")
    
    m = result['metrics']
    print("Metrics:")
    print(f"  Total Latency: {m['total_latency']:.1f}s")
    print(f"  Retrieval Time: {m['retrieval_time']:.1f}s")
    print(f"  LLM Time: {m['llm_time']:.1f}s")
    print(f"  Post-processing Time: {m['post_processing_time']:.1f}s")
    print(f"  Documents Retrieved: {m['documents_retrieved']}")
    print(f"  Documents Used in Answer: {m['documents_used_in_answer']}")
    print(f"  Input Tokens: {m['input_tokens']:,}")
    print(f"  Output Tokens: {m['output_tokens']:,}")
    print(f"  Estimated Cost: ${m['estimated_cost']:.4f}")
    print()


def print_batch_summary(results: List[Dict]):
    """Print summary for batch queries"""
    total_latency = sum(r['metrics']['total_latency'] for r in results)
    total_cost = sum(r['metrics']['estimated_cost'] for r in results)
    total_tokens = sum(r['metrics']['input_tokens'] + r['metrics']['output_tokens'] for r in results)
    
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total Questions: {len(results)}")
    print(f"Total Time: {total_latency:.2f}s")
    print(f"Average Time per Question: {total_latency/len(results):.2f}s")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Total Estimated Cost: ${total_cost:.6f}")
    print("=" * 70)


async def main(args):
    """Main entry point"""
     
    vector_store, embeddings = load_faiss_index()
    
    # Initialize query engine
    engine = AsyncQueryEngine(vector_store, embeddings)
    
    # Collect questions
    questions = []
    
    if args.question:
        questions = [args.question]
    elif args.questions:
        questions_file = Path(args.questions)
        if not questions_file.exists():
            print(f"  Questions file not found: {args.questions}")
            return
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if not questions:
        print("  No questions provided!")
        print("\nUsage:")
        print("  Single question: python query.py --question \"What is BizPay?\"")
        print("  Multiple questions: python query.py --questions questions.txt")
        print("  Concurrent processing: python query.py --questions questions.txt --concurrent")
        return
    
    # Process queries for single and batch 
    if len(questions) == 1:
 
        result = await engine.query_single(questions[0])
        print_result(result)
    else:
    
        results = await engine.query_batch(questions, concurrent=args.concurrent)
        
    
        for i, result in enumerate(results, 1):
            if i > 1:
                print("\n" + "="*70 + "\n")
            print_result(result)
        
        
        print_batch_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query the TransFi RAG system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question
  python query.py --question "What is BizPay?"
  
  # Multiple questions from file
  python query.py --questions questions.txt
  
  # Batch processing (concurrent)
  python query.py --questions questions.txt --concurrent
  
Questions file format (questions.txt):
  What is BizPay?
  How does TransFi handle international payments?
  What countries does TransFi support?
        """
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to ask'
    )
    parser.add_argument(
        '--questions',
        type=str,
        help='Path to file containing questions (one per line)'
    )
    parser.add_argument(
        '--concurrent',
        action='store_true',
        help='Process multiple questions concurrently (faster but uses more resources)'
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n\n    Interrupted by user")
    except Exception as e:
        print(f"\n  Error: {str(e)}")
        import traceback
        traceback.print_exc()