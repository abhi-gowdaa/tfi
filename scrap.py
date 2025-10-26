import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

visited = set()

def crawl(url, depth=1):
    if depth == 0 or url in visited:
        return
    visited.add(url)

    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

        print("Visiting:", url)

        
        for link in soup.find_all('a', href=True):
            new_url = urljoin(url, link['href']) 
            # print(link['href'])
            if new_url.startswith('https://www.transfi.com/products/') or new_url.startswith('https://www.transfi.com/solutions/'):
                       
                crawl(new_url, depth - 1)         

    except Exception as e:
        print("Error:", e)

 
start_url = "https://www.transfi.com/"
crawl(start_url, depth=2)
