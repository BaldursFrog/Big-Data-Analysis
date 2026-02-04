import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
from collections import Counter

class IntelligentWebCrawler:
    def __init__(self, delay=1, max_pages=50):
        self.delay = delay
        self.max_pages = max_pages
        self.visited_urls = set()
        self.data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def is_valid_url(self, url, base_domain):
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == base_domain
    
    def extract_text_content(self, soup):
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    
    def extract_links(self, soup, base_url):
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links.append(full_url)
        return links
    
    def analyze_content(self, text, url):
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        most_common_words = Counter(words).most_common(10)
        
        return {
            'url': url,
            'word_count': word_count,
            'unique_words': unique_words,
            'most_common_words': most_common_words,
            'text_sample': text[:500] + '...' if len(text) > 500 else text
        }
    
    def crawl_page(self, url, depth=0, max_depth=2):
        if (url in self.visited_urls or depth > max_depth or 
            len(self.visited_urls) >= self.max_pages):
            return []
        
        try:
            print(f"Crawling: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = self.extract_text_content(soup)
            page_data = self.analyze_content(text_content, url)
            
            page_data['title'] = soup.title.string if soup.title else 'No title'
            page_data['depth'] = depth
            page_data['status_code'] = response.status_code
            
            self.data.append(page_data)
            self.visited_urls.add(url)
            
            if depth < max_depth:
                links = self.extract_links(soup, url)
                base_domain = urlparse(url).netloc
                valid_links = [link for link in links 
                             if self.is_valid_url(link, base_domain)]
                return valid_links
            
            return []
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return []
    
    def start_crawling(self, start_url, max_depth=2):
        print(f"Starting crawl from: {start_url}")
        urls_to_visit = [start_url]
        
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url not in self.visited_urls:
                new_links = self.crawl_page(current_url, max_depth=max_depth)
                urls_to_visit.extend(new_links)
                time.sleep(self.delay)
        
        print(f"Crawling completed. Pages processed: {len(self.data)}")
        return self.data

def evaluate_data(crawled_data):
    if not crawled_data:
        return {"error": "No data to evaluate"}
    
    total_pages = len(crawled_data)
    total_words = sum(page['word_count'] for page in crawled_data)
    avg_words_per_page = total_words / total_pages if total_pages > 0 else 0
    
    high_quality_pages = [page for page in crawled_data 
                         if page['word_count'] > 100]
    
    evaluation = {
        'total_pages': total_pages,
        'total_words': total_words,
        'average_words_per_page': round(avg_words_per_page, 2),
        'high_quality_pages': len(high_quality_pages),
        'quality_ratio': f"{(len(high_quality_pages) / total_pages * 100):.1f}%",
        'unique_urls': len(set(page['url'] for page in crawled_data))
    }
    
    return evaluation