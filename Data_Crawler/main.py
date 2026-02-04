import json
import os
import time
from crawler.intelligent_crawler import IntelligentWebCrawler, evaluate_data
from crawler.data_cleaner import DataCleaner

def main():
    crawler = IntelligentWebCrawler(delay=1, max_pages=10)
    cleaner = DataCleaner()
    
    start_url = input("Enter URL to crawl: ").strip()
    
    if not start_url.startswith(('http://', 'https://')):
        start_url = 'https://' + start_url
    
    try:
        print("Starting crawler...")
        crawled_data = crawler.start_crawling(start_url, max_depth=1)
        
        print("\n--- DATA EVALUATION ---")
        evaluation = evaluate_data(crawled_data)
        for key, value in evaluation.items():
            print(f"{key}: {value}")
        
        print("\n--- DATA CLEANING ---")
        cleaned_data = cleaner.clean_crawled_data(crawled_data)
        
        timestamp = int(time.time())
        
        os.makedirs('outputs', exist_ok=True)
        
        with open(f'outputs/crawled_data_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(crawled_data, f, ensure_ascii=False, indent=2)
        
        with open(f'outputs/cleaned_data_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        with open(f'outputs/evaluation_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to files:")
        print(f"- outputs/crawled_data_{timestamp}.json (raw data)")
        print(f"- outputs/cleaned_data_{timestamp}.json (cleaned data)")
        print(f"- outputs/evaluation_{timestamp}.json (evaluation)")
        
        if cleaned_data:
            print(f"\n--- CLEANED DATA SAMPLE ---")
            sample = cleaned_data[0]
            print(f"URL: {sample['url']}")
            print(f"Title: {sample.get('cleaned_title', 'N/A')}")
            print(f"Text (first 200 chars): {sample.get('cleaned_text', 'N/A')[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()