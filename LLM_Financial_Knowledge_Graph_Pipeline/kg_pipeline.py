import os
import re
import json
import requests
from py2neo import Graph, Node, Relationship, NodeMatcher
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import spacy
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_API_URL

class KnowledgeGraphPipeline:
    def __init__(self):
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.matcher = NodeMatcher(self.graph)
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt', quiet=True)
        
    def preprocess_text(self, raw_text):
        text = re.sub(r'<[^>]+>', '', raw_text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        processed_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                processed_sentences.append(sentence.strip())
        return processed_sentences
    
    def process_financial_news_json(self, json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        processed_articles = []
        for article in news_data:
            processed_content = self.preprocess_text(article.get('content', ''))
            processed_article = {
                'id': article.get('id'),
                'date': article.get('date'),
                'title': article.get('title'),
                'source': article.get('source'),
                'processed_sentences': processed_content
            }
            processed_articles.append(processed_article)
        
        return processed_articles
    
    def process_financial_news_txt(self, txt_file_path):
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        articles = content.split('-' * 80)
        processed_articles = []
        
        for article in articles:
            if not article.strip():
                continue
                
            lines = article.strip().split('\n')
            article_data = {}
            
            for line in lines:
                if line.startswith('Date:'):
                    article_data['date'] = line.replace('Date:', '').strip()
                elif line.startswith('Title:'):
                    article_data['title'] = line.replace('Title:', '').strip()
                elif line.startswith('Source:'):
                    article_data['source'] = line.replace('Source:', '').strip()
                elif line.startswith('Content:'):
                    content_text = line.replace('Content:', '').strip()
                    article_data['processed_sentences'] = self.preprocess_text(content_text)
            
            if 'processed_sentences' in article_data:
                processed_articles.append(article_data)
        
        return processed_articles
    
    def extract_entities_relationships(self, text):
        zero_shot_prompt = f"""
You are a financial information extraction expert. Extract entities and relationships from the following financial news text.

Entity Types to extract:
- COMPANY: Business organizations (Apple, Microsoft, Tesla, etc.)
- PERSON: Executives, CEOs, analysts (Tim Cook, Satya Nadella, etc.)
- PRODUCT: Products or services (iPhone, Windows, Cloud Services, etc.)
- FINANCIAL_METRIC: Financial terms (revenue, profit margin, market cap, etc.)
- SECTOR: Industry sectors (Technology, Healthcare, Semiconductors, etc.)

Relationship Types to extract:
- ACQUIRES: Company acquisitions and mergers
- PARTNERS_WITH: Strategic partnerships
- INVESTS_IN: Investment activities
- ANNOUNCES: Company announcements
- REPORTS: Financial reporting
- INCREASES/DECREASES: Metric changes
- LAUNCHES: Product or service launches
- PROJECTS: Future projections

Text: {text}

Extract only the most important and clear relationships. Return as a JSON array:
[
    {{"subject": "Apple", "predicate": "ACQUIRES", "object": "TechCorp"}},
    {{"subject": "Apple", "predicate": "REPORTS", "object": "revenue growth"}}
]
"""

        few_shot_prompt = f"""
You are a financial information extraction expert. Extract entities and relationships from financial news text.

Entity Types: COMPANY, PERSON, PRODUCT, FINANCIAL_METRIC, SECTOR
Relationship Types: ACQUIRES, PARTNERS_WITH, INVESTS_IN, ANNOUNCES, REPORTS, INCREASES, DECREASES, LAUNCHES, PROJECTS

Example 1:
Text: "Apple announced today that it will acquire TechCorp for $1 billion in a deal that will strengthen their position in the technology market."
Output: [
    {{"subject": "Apple", "predicate": "ACQUIRES", "object": "TechCorp"}},
    {{"subject": "Apple", "predicate": "ANNOUNCES", "object": "acquisition of TechCorp"}}
]

Example 2:
Text: "Microsoft reported strong quarterly earnings with a 15% increase in cloud services revenue."
Output: [
    {{"subject": "Microsoft", "predicate": "REPORTS", "object": "quarterly earnings"}},
    {{"subject": "Microsoft", "predicate": "INCREASES", "object": "cloud services revenue"}}
]

Example 3:
Text: "Tesla and Panasonic partnered to develop new battery technology for electric vehicles."
Output: [
    {{"subject": "Tesla", "predicate": "PARTNERS_WITH", "object": "Panasonic"}},
    {{"subject": "Tesla", "predicate": "DEVELOPS", "object": "battery technology"}}
]

Now extract from this text:
Text: {text}

Output:
"""
        
        prompt = few_shot_prompt
        
        response = requests.post(
            LLM_API_URL,
            json={"prompt": prompt, "max_tokens": 800}
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                extracted_text = result.get("text", "[]")
                if extracted_text.startswith("Output:"):
                    extracted_text = extracted_text[7:].strip()
                return json.loads(extracted_text)
            except json.JSONDecodeError:
                try:
                    extracted_text = result.get("text", "[]")
                    start_idx = extracted_text.find('[')
                    end_idx = extracted_text.rfind(']') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = extracted_text[start_idx:end_idx]
                        return json.loads(json_str)
                except:
                    return []
            except Exception as e:
                print(f"Error in extraction: {e}")
                return []
        return []
    
    def standardize_entity_name(self, entity_name):
        if not entity_name or not isinstance(entity_name, str):
            return ""
        
        entity_name = entity_name.strip()
        
        # Remove common legal suffixes
        suffixes = [" Inc.", " Inc", " Corporation", " Corp.", " Corp", " LLC", " Ltd.", " Ltd"]
        for suffix in suffixes:
            if entity_name.endswith(suffix):
                entity_name = entity_name[:-len(suffix)].strip()
        
        # Handle special cases
        if entity_name.lower() in ["apple", "apple inc."]:
            return "Apple"
        elif entity_name.lower() in ["microsoft", "microsoft corporation"]:
            return "Microsoft"
        elif entity_name.lower() in ["tesla", "tesla inc."]:
            return "Tesla"
        
        # Capitalize properly
        return entity_name
    
    def validate_relationship_type(self, predicate):
        if not predicate or not isinstance(predicate, str):
            return ""
        
        predicate = predicate.strip().upper()
        
        # List of valid relationship types
        valid_relationships = [
            "ACQUIRES", "PARTNERS_WITH", "INVESTS_IN", "ANNOUNCES", "REPORTS",
            "INCREASES", "DECREASES", "LAUNCHES", "PROJECTS", "DEVELOPS",
            "PRODUCES", "FOCUSES_ON", "PROPOSES", "CONCERNS"
        ]
        
        # Check if the predicate is valid
        if predicate in valid_relationships:
            return predicate
        
        # Handle common variations
        if "ACQUIRE" in predicate:
            return "ACQUIRES"
        elif "PARTNER" in predicate:
            return "PARTNERS_WITH"
        elif "INVEST" in predicate:
            return "INVESTS_IN"
        elif "ANNOUNCE" in predicate:
            return "ANNOUNCES"
        elif "REPORT" in predicate:
            return "REPORTS"
        elif "INCREASE" in predicate:
            return "INCREASES"
        elif "DECREASE" in predicate:
            return "DECREASES"
        elif "LAUNCH" in predicate:
            return "LAUNCHES"
        elif "PROJECT" in predicate:
            return "PROJECTS"
        elif "DEVELOP" in predicate:
            return "DEVELOPS"
        elif "PRODUCE" in predicate:
            return "PRODUCES"
        elif "FOCUS" in predicate:
            return "FOCUSES_ON"
        elif "PROPOSE" in predicate:
            return "PROPOSES"
        elif "CONCERN" in predicate:
            return "CONCERNS"
        
        # If not valid, return empty string
        return ""
    
    def remove_duplicate_triplets(self, triplets):
        seen = set()
        unique_triplets = []
        
        for triplet in triplets:
            # Create a normalized representation for comparison
            key = (triplet["subject"].lower(), triplet["predicate"], triplet["object"].lower())
            
            if key not in seen:
                seen.add(key)
                unique_triplets.append(triplet)
        
        return unique_triplets
    
    def correct_relationship_direction(self, triplet):
        subject = triplet["subject"]
        predicate = triplet["predicate"]
        object = triplet["object"]
        
        # Define relationships that typically flow from company to entity
        company_to_entity = ["ACQUIRES", "INVESTS_IN", "LAUNCHES", "PRODUCES", "ANNOUNCES"]
        
        # Define relationships that typically flow from entity to company
        entity_to_company = ["REPORTS", "INCREASES", "DECREASES"]
        
        # Simple heuristic: if subject looks like a company and predicate is in entity_to_company
        # or object looks like a company and predicate is in company_to_entity
        # we might need to flip the relationship
        
        # This is a simplified heuristic - in a real system, you'd use more sophisticated NLP
        return triplet
    
    def clean_triplets(self, raw_triplets):
        cleaned = []
        
        for triplet in raw_triplets:
            if not isinstance(triplet, dict):
                continue
                
            if "subject" not in triplet or "predicate" not in triplet or "object" not in triplet:
                continue
            
            # Extract and clean components
            subject = self.standardize_entity_name(triplet["subject"])
            predicate = self.validate_relationship_type(triplet["predicate"])
            object = self.standardize_entity_name(triplet["object"])
            
            # Skip if any component is invalid
            if not subject or not predicate or not object:
                continue
            
            # Create cleaned triplet
            cleaned_triplet = {
                "subject": subject,
                "predicate": predicate,
                "object": object
            }
            
            # Apply relationship direction correction
            cleaned_triplet = self.correct_relationship_direction(cleaned_triplet)
            
            cleaned.append(cleaned_triplet)
        
        # Remove duplicates
        cleaned = self.remove_duplicate_triplets(cleaned)
        
        return cleaned
    
    def determine_entity_type(self, entity_name):
        entity_name_lower = entity_name.lower()
        
        # Known companies
        known_companies = ["apple", "microsoft", "tesla", "google", "amazon", "facebook", "netflix", "panasonic", "techcorp"]
        if entity_name_lower in known_companies:
            return "Company"
        
        # Company indicators
        company_indicators = ["inc", "corp", "corporation", "llc", "ltd", "company", "technologies", "systems"]
        if any(indicator in entity_name_lower for indicator in company_indicators):
            return "Company"
        
        # Known persons
        known_persons = ["tim cook", "satya nadella", "elon musk"]
        if entity_name_lower in known_persons:
            return "Person"
        
        # Person indicators (simple heuristic) - check if it's a name with spaces
        if len(entity_name.split()) == 2 and entity_name.istitle():
            return "Person"
        
        # Product indicators
        product_indicators = ["iphone", "windows", "android", "model", "version", "service", "platform"]
        if any(indicator in entity_name_lower for indicator in product_indicators):
            return "Product"
        
        # Financial metric indicators
        metric_indicators = ["revenue", "profit", "earnings", "growth", "margin", "market cap", "sales", "quarterly earnings"]
        if any(indicator in entity_name_lower for indicator in metric_indicators):
            return "FinancialMetric"
        
        # Sector indicators
        sector_indicators = ["technology", "healthcare", "finance", "automotive", "retail", "energy"]
        if entity_name_lower in sector_indicators:
            return "Sector"
        
        # Default to Entity
        return "Entity"
    
    def build_graph(self, cleaned_triplets):
        # Clear existing graph for clean visualization
        self.graph.run("MATCH (n) DETACH DELETE n")
        
        # Create nodes with proper labels using MERGE
        entities = set()
        for triplet in cleaned_triplets:
            entities.add(triplet["subject"])
            entities.add(triplet["object"])
        
        # Create nodes with appropriate labels
        for entity in entities:
            entity_type = self.determine_entity_type(entity)
            query = f"MERGE (n:{entity_type} {{name: $name}})"
            self.graph.run(query, name=entity)
        
        # Create relationships using MERGE to avoid duplicates
        for triplet in cleaned_triplets:
            subject = triplet["subject"]
            predicate = triplet["predicate"]
            object = triplet["object"]
            
            subject_type = self.determine_entity_type(subject)
            object_type = self.determine_entity_type(object)
            
            query = f"""
            MATCH (a:{subject_type} {{name: $subject}})
            MATCH (b:{object_type} {{name: $object}})
            MERGE (a)-[r:{predicate}]->(b)
            """
            
            self.graph.run(query, subject=subject, object=object)
        
        # Create visualization-friendly indexes
        self.graph.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n) ON (n.name)")
        
        # Set up basic visualization properties
        self.graph.run("""
        MATCH (n:Company)
        SET n.color = '#ff9900'
        """)
        
        self.graph.run("""
        MATCH (n:Person)
        SET n.color = '#66ccff'
        """)
        
        self.graph.run("""
        MATCH (n:Product)
        SET n.color = '#99ff99'
        """)
        
        self.graph.run("""
        MATCH (n:FinancialMetric)
        SET n.color = '#ff6666'
        """)
        
        self.graph.run("""
        MATCH (n:Sector)
        SET n.color = '#ffccff'
        """)
        
        self.graph.run("""
        MATCH (n:Entity)
        WHERE NOT exists(n.color)
        SET n.color = '#cccccc'
        """)
    
    def nl_to_cypher(self, natural_language_query):
        schema_info = """
        Graph Schema:
        Node Types: Company, Person, Product, FinancialMetric, Sector, Entity
        Relationship Types: ACQUIRES, PARTNERS_WITH, INVESTS_IN, ANNOUNCES, REPORTS, INCREASES, DECREASES, LAUNCHES, PROJECTS, DEVELOPS, PRODUCES, FOCUSES_ON, PROPOSES, CONCERNS
        All nodes have a 'name' property
        """
        
        few_shot_examples = """
        Example 1:
        Question: Which companies has Apple acquired?
        Cypher: MATCH (c1:Company {name: "Apple"})-[:ACQUIRES]->(c2:Company) RETURN c1.name as Acquirer, c2.name as Acquired
        
        Example 2:
        Question: What products has Tesla launched?
        Cypher: MATCH (c:Company {name: "Tesla"})-[:LAUNCHES]->(p:Product) RETURN c.name as Company, p.name as Product
        
        Example 3:
        Question: Who is the CEO of Microsoft?
        Cypher: MATCH (c:Company {name: "Microsoft"})-[:HAS_CEO]->(p:Person) RETURN c.name as Company, p.name as CEO
        
        Example 4:
        Question: What financial metrics has Apple reported?
        Cypher: MATCH (c:Company {name: "Apple"})-[:REPORTS]->(f:FinancialMetric) RETURN c.name as Company, f.name as Metric
        
        Example 5:
        Question: Show me all partnerships between companies
        Cypher: MATCH (c1:Company)-[:PARTNERS_WITH]->(c2:Company) RETURN c1.name as Company1, c2.name as Company2
        
        Example 6:
        Question: What has Microsoft announced recently?
        Cypher: MATCH (c:Company {name: "Microsoft"})-[:ANNOUNCES]->(e) RETURN c.name as Company, e.name as Announcement
        """
        
        prompt = f"""
        You are a Cypher query expert for a financial knowledge graph. Convert natural language questions to valid Cypher queries.
        
        {schema_info}
        
        {few_shot_examples}
        
        Rules:
        1. Use exact node labels from the schema
        2. Use exact relationship types from the schema
        3. Always use the 'name' property for matching entities
        4. Return meaningful aliases for results
        5. Only return the Cypher query, no explanations
        
        Question: {natural_language_query}
        
        Cypher:
        """
        
        response = requests.post(
            LLM_API_URL,
            json={"prompt": prompt, "max_tokens": 300}
        )
        
        if response.status_code == 200:
            result = response.json()
            cypher_query = result.get("text", "").strip()
            
            # Clean up the response if it contains extra text
            if cypher_query.startswith("Cypher:"):
                cypher_query = cypher_query[7:].strip()
            
            # Remove any trailing explanations
            if "\n" in cypher_query:
                cypher_query = cypher_query.split("\n")[0].strip()
            
            return cypher_query
        return ""
    
    def execute_query(self, cypher_query):
        try:
            result = self.graph.run(cypher_query)
            records = [dict(record) for record in result]
            return records
        except Exception as e:
            print(f"Query execution error: {e}")
            return []
    
    def format_results(self, results):
        if not results:
            return "No results found."
        
        formatted_output = []
        for i, record in enumerate(results, 1):
            formatted_record = f"Result {i}:\n"
            for key, value in record.items():
                if hasattr(value, 'get'):
                    # Handle Neo4j Node/Relationship objects
                    if hasattr(value, 'labels'):
                        node_type = ','.join(value.labels)
                        node_name = value.get('name', 'Unknown')
                        formatted_record += f"  {key}: {node_type} - {node_name}\n"
                    else:
                        formatted_record += f"  {key}: {value}\n"
                else:
                    formatted_record += f"  {key}: {value}\n"
            formatted_output.append(formatted_record)
        
        return "\n".join(formatted_output)
    
    def process_news_file(self, file_path):
        if file_path.endswith('.json'):
            processed_articles = self.process_financial_news_json(file_path)
        elif file_path.endswith('.txt'):
            processed_articles = self.process_financial_news_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .json or .txt files.")
        
        all_triplets = []
        for article in processed_articles:
            for sentence in article.get('processed_sentences', []):
                raw_triplets = self.extract_entities_relationships(sentence)
                cleaned_triplets = self.clean_triplets(raw_triplets)
                all_triplets.extend(cleaned_triplets)
        
        self.build_graph(all_triplets)
        return all_triplets, processed_articles
    
    def query_interface(self):
        print("Financial Knowledge Graph Query Interface")
        print("Enter 'quit' to exit")
        print("Example queries:")
        print("- Which companies has Apple acquired?")
        print("- What products has Tesla launched?")
        print("- Show me all partnerships between companies")
        print("- What financial metrics has Microsoft reported?")
        print()
        
        while True:
            query = input("Enter your question: ").strip()
            if query.lower() == 'quit':
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            print("\nGenerating Cypher query...")
            cypher_query = self.nl_to_cypher(query)
            
            if cypher_query:
                print(f"Generated Cypher: {cypher_query}")
                print("\nExecuting query...")
                results = self.execute_query(cypher_query)
                formatted_results = self.format_results(results)
                print("\nResults:")
                print(formatted_results)
                print("-" * 50)
            else:
                print("Could not generate a valid query. Please try rephrasing your question.")
                print("-" * 50)

if __name__ == "__main__":
    pipeline = KnowledgeGraphPipeline()
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    json_file = os.path.join(data_dir, "financial_news.json")
    txt_file = os.path.join(data_dir, "financial_news.txt")
    
    if os.path.exists(json_file):
        print(f"Processing JSON file: {json_file}")
        triplets, processed_articles = pipeline.process_news_file(json_file)
        print(f"Processed {len(processed_articles)} articles")
        print(f"Extracted {len(triplets)} triplets")
    elif os.path.exists(txt_file):
        print(f"Processing TXT file: {txt_file}")
        triplets, processed_articles = pipeline.process_news_file(txt_file)
        print(f"Processed {len(processed_articles)} articles")
        print(f"Extracted {len(triplets)} triplets")
    else:
        sample_file = os.path.join(data_dir, "sample_news.txt")
        if not os.path.exists(sample_file):
            with open(sample_file, 'w') as f:
                f.write("Apple announced today that it will acquire TechCorp for $1 billion. "
                       "The deal is expected to close next quarter. Apple CEO Tim Cook said this "
                       "acquisition will strengthen their position in the technology market.")
        
        triplets, processed_articles = pipeline.process_news_file(sample_file)
        print(f"Extracted {len(triplets)} triplets")
    
    pipeline.query_interface()