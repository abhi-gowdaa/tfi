import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import json
import time
from typing import List, Dict, Set
import hashlib


class AsyncTransFiScraper:
    def __init__(self, base_url: str = "https://www.transfi.com"):
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        self.failed_urls: List[str] = []
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'start_time': 0,
            'end_time': 0,
            'pages_scraped': 0,
            'pages_failed': 0,
            'scraping_times': []
        }
    
    def url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        parsed = urlparse(url)
        path = parsed.path.strip('/').replace('/', '_')
        return f"{path}_{url_hash}" if path else f"index_{url_hash}"
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> tuple:
        """Fetch a single page asynchronously"""
        if url in self.visited_urls:
            return None, None
        
        start_time = time.time()
        self.visited_urls.add(url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            async with session.get(url, headers=headers, timeout=15) as response:
                html = await response.text()
                elapsed = time.time() - start_time
                self.metrics['scraping_times'].append(elapsed)
                
                # Save raw HTML
                filename = self.url_to_filename(url)
                with open(self.raw_dir / f"{filename}.html", 'w', encoding='utf-8') as f:
                    f.write(html)
                
                return html, elapsed
                
        except Exception as e:
            self.failed_urls.append(url)
            self.metrics['pages_failed'] += 1
            print(f"  Error fetching {url}: {str(e)}")
            return None, None
    
    def determine_category(self, url: str) -> str:
        """Determine the category based on URL path"""
        if '/products/' in url:
            return 'product'
        elif '/solutions/' in url:
            return 'solution'
        elif '/supported' in url:
            return 'supported'
        else:
            return 'other'
    
    def parse_product_page(self, soup: BeautifulSoup, url: str) -> Dict:
        """Parse a page and extract structured data"""
        data = {
            'url': url,
            'title': '',
            'short_description': '',
            'long_description': '',
            'category': self.determine_category(url)
        }
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
            tag.decompose()
        
        # Extract title
        title_selectors = [
            'h1.banner-title-new-2',
            'h1.home-new-banner-title',
            'h1.page-title',
            'h1'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                data['title'] = title_elem.get_text(strip=True)
                break
        
        if not data['title']:
            title_tag = soup.find('title')
            if title_tag:
                data['title'] = title_tag.get_text(strip=True).split('|')[0].strip()
        
        # Extract short description
        short_desc_selectors = [
            'p.home-banner-content',
            'p.features_hero_para',
            '.product_hero_para p',
            'p.banner-content',
            'meta[name="description"]'
        ]
        
        for selector in short_desc_selectors:
            if selector.startswith('meta'):
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    data['short_description'] = meta_desc['content']
                    break
            else:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    data['short_description'] = desc_elem.get_text(strip=True)
                    break
        
        # Extract long description
        content_parts = []
        
        # For supported pages, extract tables and lists
        if data['category'] == 'supported':
            # Extract table data
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = ' | '.join([cell.get_text(strip=True) for cell in cells])
                        if row_text:
                            content_parts.append(row_text)
            
            # Extract list items
            lists = soup.find_all(['ul', 'ol'])
            for lst in lists:
                items = lst.find_all('li')
                for item in items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 10:
                        content_parts.append(text)
        
        # Extract from main sections
        main_sections = soup.find_all('section')
        
        for section in main_sections:
            section_classes = ' '.join(section.get('class', []))
            if any(skip in section_classes for skip in ['footer', 'nav', 'header', 'navbar']):
                continue
            
            for elem in section.find_all(['h2', 'h3', 'h4', 'h5', 'p', 'li', 'td']):
                parent_classes = ' '.join(elem.parent.get('class', []))
                if any(skip in parent_classes for skip in ['button', 'btn', 'nav', 'menu']):
                    continue
                
                text = elem.get_text(strip=True)
                
                # Filter out unwanted text
                if (text and len(text) > 20 and 
                    not text.startswith('http') and
                    'Contact Sales' not in text and 
                    'Sign Up' not in text and
                    'Learn More' not in text and
                    'Get Started' not in text):
                    content_parts.append(text)
        
        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for part in content_parts:
            if part not in seen and len(part) > 20:
                seen.add(part)
                unique_parts.append(part)
        
        data['long_description'] = '\n\n'.join(unique_parts)
        
        return data
    
    def extract_product_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract all relevant product, solution, and supported page links"""
        links = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            
            # Only include pages that match our criteria
            if self.base_url in full_url:
                parsed = urlparse(full_url)
                path = parsed.path
                
                # Check if it's a valid page type
                is_product = '/products/' in path
                is_solution = '/solutions/' in path
                # Only include /supported pages directly under .com (not language-specific like /id/supported)
                is_supported = path.startswith('/supported') and not path.startswith('/supported/')
                
                # Alternative check: path matches /supported-* pattern
                is_supported = is_supported or (path.startswith('/supported-') and '/' not in path[1:].split('supported')[1])
                
                if is_product or is_solution or is_supported:
                    # Clean URL (remove fragments and query params for consistency)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{path}"
                    links.add(clean_url)
        
        return links
    
    async def scrape_url(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Scrape a single URL and parse its content"""
        html, elapsed = await self.fetch_page(session, url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        product_data = self.parse_product_page(soup, url)
        
        self.metrics['pages_scraped'] += 1
        category = product_data['category']
        if category=='supported':
            print(f"  Scraped [{self.metrics['pages_scraped']}] ({category}): {url} ({elapsed:.2f}s)")
        
        return product_data
    
    async def discover_all_links(self, session: aiohttp.ClientSession, start_url: str) -> Set[str]:
        """
        Discover all relevant links by crawling main pages and navigation.
        This ensures we get all /supported pages even if not directly linked.
        """
        all_links = set()
        
        # Pages to check for links
        discovery_urls = [
            start_url,
            f"{self.base_url}/products",
            f"{self.base_url}/solutions",
            f"{self.base_url}/supported-countries-payment-methods",
        ]
        
        print(f"🔍 Discovering links from {len(discovery_urls)} navigation pages...")
        
        for url in discovery_urls:
            html, _ = await self.fetch_page(session, url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                links = self.extract_product_links(soup, url)
                all_links.update(links)
                print(f"  Found {len(links)} links from {url}")
        
        return all_links
    
    async def scrape_all(self, start_url: str) -> List[Dict]:
        """Main scraping orchestration"""
        self.metrics['start_time'] = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Discover all relevant links
            product_links = await self.discover_all_links(session, start_url)
            
            print(f"\n Total unique pages to scrape: {len(product_links)}")
            print(f"   - Products: {len([l for l in product_links if '/products/' in l])}")
            print(f"   - Solutions: {len([l for l in product_links if '/solutions/' in l])}")
            print(f"   - Supported: {len([l for l in product_links if '/supported' in l])}")
            print()
            
            # Scrape all pages concurrently
            tasks = [self.scrape_url(session, link) for link in product_links]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None and exceptions
            self.scraped_data = [r for r in results if r and isinstance(r, dict)]
        
        self.metrics['end_time'] = time.time()
        
        # Save processed data
        output_file = self.processed_dir / "transfi_products.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Saved {len(self.scraped_data)} documents to {output_file}")
        
        return self.scraped_data
    
    def get_metrics(self) -> Dict:
        """Calculate and return comprehensive metrics"""
        total_time = self.metrics['end_time'] - self.metrics['start_time']
        avg_time = sum(self.metrics['scraping_times']) / len(self.metrics['scraping_times']) if self.metrics['scraping_times'] else 0
        
        return {
            'total_time': f"{total_time:.1f}s",
            'pages_scraped': self.metrics['pages_scraped'],
            'pages_failed': self.metrics['pages_failed'],
            'average_scraping_time': f"{avg_time:.2f}s",
            'failed_urls': self.failed_urls
        }