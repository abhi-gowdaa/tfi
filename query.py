import asyncio
import argparse
import time
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from core.retrival import AsyncRetriever

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
        print(f"Creating new FAISS index at: {INDEX_PATH}")
        vector_store = FAISS.from_texts(["init"], embedding=embeddings)
        return vector_store, embeddings


class AsyncQueryEngine:
    """Wrapper around AsyncRetriever for CLI usage"""
    
    def __init__(self, vector_store=None, embeddings=None):
        # Initialize AsyncRetriever with the index
        self.retriever = AsyncRetriever(index_path=INDEX_PATH)
        
        # If vector_store is provided (from API), update the retriever's vector store
        if vector_store:
            self.retriever.vector_store = vector_store
    
    async def query_single(self, question: str, k: int = 5) -> Dict:
        """
        Process a single question - delegates to AsyncRetriever
        """
        return await self.retriever.query(question, k)
    
    async def query_batch(self, questions: List[str], concurrent: bool = False) -> List[Dict]:
        """
        Process multiple questions - delegates to AsyncRetriever
        """
        return await self.retriever.query_batch(questions, k=5, concurrent=concurrent)


def print_result(result: Dict, show_header: bool = True):
    """Print a single query result"""
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    
    print("Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['title']}")
        print(f"     URL: {source['url']}")
        print(f"     Snippet: \"{source['snippet']}\"")
        print(f"     Relevance Score: {source['relevance_score']:.4f}")
    
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
    print("Loading FAISS index...")
    
    # Initialize query engine (will load index internally via AsyncRetriever)
    engine = AsyncQueryEngine()
    
    # Collect questions
    questions = []
    
    if args.question:
        questions = [args.question]
    elif args.questions:
        questions_file = Path(args.questions)
        if not questions_file.exists():
            print(f"Questions file not found: {args.questions}")
            return
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if not questions:
        print("No questions provided!")
        print("\nUsage:")
        print("  Single question: python query.py --question \"What is BizPay?\"")
        print("  Multiple questions: python query.py --questions questions.txt")
        print("  Concurrent processing: python query.py --questions questions.txt --concurrent")
        return
    
    # Process queries
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
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()