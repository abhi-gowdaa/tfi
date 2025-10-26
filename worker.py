from datetime import datetime
from Service.DBService.url_service import UrlService
from Service.embedding import chunk_text, scrape_url, store_embedding
 
def loop(smallest,largest):
    while smallest <= largest:
        print(smallest)
        smallest += 1
    return largest

 
#job
def process_ingest_url(url:str,url_id:str):
    try:
        url_service=UrlService()
        url_service.update_status(url_id, status="processing")
        
        scraped_text = scrape_url(url)
        
        if not scraped_text:
            raise ValueError("No text content scraped from URL")
        
        chunks=chunk_text(scraped_text,chunk_size=1000, overlap=200)
        store_embedding(chunks, url, url_id)
        
        
        url_service.update_status(
            url_id=url_id,
            status="completed",
        )
    
        return {
            "success": True,
            "url_id": url_id,
            "url": url,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        error_message = str(e)
        
        try:
            url_service.update_status(
                url_id, 
                status="failed",
                error=error_message
            )
        except Exception:
            pass
        
        raise
        
        

