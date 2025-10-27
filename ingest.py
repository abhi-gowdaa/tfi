import asyncio
import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.scraper import AsyncTransFiScraper
from core.embedder import AsyncEmbedder
import json


async def main(url: str):
    print("=" * 60)
    print(" Starting TransFi RAG Ingestion Pipeline")
    print("=" * 60)
    
    total_start = time.time()
    
    # Step 1: Scraping
    print("\nPhase 1: Scraping TransFi website...")
    scraper = AsyncTransFiScraper()
    
    try:
        documents = await scraper.scrape_all(url)
        scraper_metrics = scraper.get_metrics()
        
        if not documents:
            print("  No documents scraped. Exiting.")
            return
        
        print(f"\n  Scraped {len(documents)} products/solutions")
        
        # Step 2: Chunking & Embedding
        print("\n Phase 2: Chunking and Embedding...")
        embedder = AsyncEmbedder()
        await embedder.process_documents(documents)
        embedder_metrics = embedder.get_metrics()
        
        total_time = time.time() - total_start
        
        # Print comprehensive metrics (only once)
        print("\n" + "=" * 60)
        print("=== INGESTION METRICS ===")
        print("=" * 60)
        print(f"Total Time: {total_time:.1f}s")
        print(f"Pages Scraped: {scraper_metrics['pages_scraped']}")
        print(f"Pages Failed: {scraper_metrics['pages_failed']}")
        print(f"Total Chunks Created: {embedder_metrics['chunks_created']}")
        print(f"Total Tokens Processed: {embedder_metrics['tokens_processed']}")
        print(f"Embedding Generation Time: {embedder_metrics['embedding_time']}")
        print(f"Indexing Time: {embedder_metrics['indexing_time']}")
        print(f"Average Scraping Time per Page: {scraper_metrics['average_scraping_time']}")
        
        if scraper_metrics['failed_urls']:
            print(f"\n    Errors: {len(scraper_metrics['failed_urls'])} URLs failed")
            for failed_url in scraper_metrics['failed_urls'][:5]:
                print(f"  - {failed_url}")
        else:
            print("\n  No errors encountered!")
        
        print("=" * 60)
        
        # Save metrics to file
        metrics_data = {
            'total_time': total_time,
            'scraper': scraper_metrics,
            'embedder': embedder_metrics,
            'documents_processed': len(documents)
        }
        
        Path('data').mkdir(exist_ok=True)
        with open('data/ingestion_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print("\n Metrics saved to data/ingestion_metrics.json")
        
    except Exception as e:
        print(f"\n  Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Ingest TransFi website content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py
  python ingest.py --url https://www.transfi.com
  python ingest.py --url "https://www.transfi.com"
        """
    )
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='URL to scrape (required)'
    )
    
    args = parser.parse_args()
    
    print(f"\n Target URL: {args.url}\n")
    
    # Run the async main function
    asyncio.run(main(args.url))