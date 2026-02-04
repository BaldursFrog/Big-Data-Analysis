import re

class DataCleaner:
    def __init__(self):
        self.cleaning_rules = {
            'remove_extra_spaces': True,
            'remove_special_chars': False,
            'min_word_length': 2,
            'remove_stopwords': True
        }
        
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'
        }
    
    def clean_text(self, text):
        if not text:
            return ""
        
        cleaned_text = text
        
        if self.cleaning_rules['remove_extra_spaces']:
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if self.cleaning_rules['remove_special_chars']:
            cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        
        if (self.cleaning_rules['min_word_length'] > 1 or 
            self.cleaning_rules['remove_stopwords']):
            
            words = cleaned_text.split()
            filtered_words = []
            
            for word in words:
                if len(word) >= self.cleaning_rules['min_word_length']:
                    if (not self.cleaning_rules['remove_stopwords'] or 
                        word.lower() not in self.stop_words):
                        filtered_words.append(word)
            
            cleaned_text = ' '.join(filtered_words)
        
        return cleaned_text
    
    def clean_crawled_data(self, crawled_data):
        cleaned_data = []
        
        for page_data in crawled_data:
            cleaned_page = page_data.copy()
            
            if 'text_sample' in cleaned_page:
                cleaned_page['cleaned_text'] = self.clean_text(
                    cleaned_page['text_sample']
                )
            
            if 'title' in cleaned_page:
                cleaned_page['cleaned_title'] = self.clean_text(
                    cleaned_page['title']
                )
            
            cleaned_data.append(cleaned_page)
        
        return cleaned_data
    
    def set_cleaning_rules(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.cleaning_rules:
                self.cleaning_rules[key] = value