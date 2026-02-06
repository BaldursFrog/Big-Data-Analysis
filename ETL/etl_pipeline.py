import PyPDF2
import spacy
import json
import os

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract raw text from PDF."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_entities(text):
    """Use spaCy to extract entities like dates, money, organizations."""
    doc = nlp(text)
    entities = {"dates": [], "money": [], "orgs": [], "persons": []}
    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["money"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["orgs"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
    return entities

def ai_extract_data(text, topic):
    """Simulate AI extraction (replace with actual AI call if available)."""
    facts = []
    figures = []
    comments = []
    if "revenue" in topic.lower():
        if "amazon" in text.lower():
            facts.append("Amazon reported significant revenue growth.")
            figures.append("$574.8 billion in net sales for 2023.")
            comments.append("This represents a 14% increase from the previous year.")
    return {"facts": facts, "figures": figures, "comments": comments}

def clean_data(data):
    """Clean and structure the extracted data."""
    cleaned = {}
    for key in data:
        if isinstance(data[key], list):
            cleaned[key] = list(set(data[key]))  # Remove duplicates
        else:
            cleaned[key] = data[key]
    return cleaned

def intelligent_crawler(input_text):
    """Intelligent crawler program for data extraction."""
    topic = input_text
    pdf_text = extract_text_from_pdf("Amazon-2024-Annual-Report.pdf")
    entities = extract_entities(pdf_text)
    ai_data = ai_extract_data(pdf_text, topic)
    combined = {**entities, **ai_data}
    return combined

def cleaning_program(data):
    """Cleaning program for data."""
    cleaned = clean_data(data)
    return cleaned

def run_etl_pipeline(topic):
    """Run the full ETL pipeline."""
    pdf_text = extract_text_from_pdf("Amazon-2024-Annual-Report.pdf")
    entities = extract_entities(pdf_text)
    ai_data = ai_extract_data(pdf_text, topic)

    raw_data = {**entities, **ai_data}
    cleaned_data = cleaning_program(raw_data)

    with open("extracted_data.json", "w") as f:
        json.dump(cleaned_data, f, indent=2)

    return cleaned_data

if __name__ == "__main__":
    topic = "Amazon's revenue growth"
    result = run_etl_pipeline(topic)
    print("ETL Pipeline Result:", json.dumps(result, indent=2))