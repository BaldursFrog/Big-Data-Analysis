
# Financial Knowledge Graph Construction with LLM and Neo4j

## Overview

This project implements a **pipeline for constructing a knowledge graph from financial news data** using Large Language Models (LLM) and the Neo4j graph database.

The system processes unstructured financial news articles, extracts entities and relationships, cleans and validates the extracted data, and builds a structured **knowledge graph** that can be queried using natural language.

The pipeline demonstrates how modern NLP and graph databases can be combined to transform raw textual data into structured knowledge representations.

---

# Project Architecture

The system consists of several main components:

1. **Data preprocessing**
2. **Entity and relationship extraction**
3. **Data cleaning and normalization**
4. **Knowledge graph construction**
5. **Natural language query interface**

Data flow of the system:

```
Raw financial news
        ↓
Text preprocessing
        ↓
LLM entity & relation extraction
        ↓
Triplet cleaning and validation
        ↓
Knowledge graph construction (Neo4j)
        ↓
Natural language → Cypher queries
        ↓
Query results
```

---

# Technology Stack

* **Python** – core implementation
* **Neo4j** – graph database
* **Py2neo** – Neo4j Python driver
* **spaCy / NLTK** – text processing
* **LLM API** – entity and relation extraction
* **Cypher** – graph query language

---

# Project Structure

```
.
├── data
│   ├── financial_news.json
│   └── financial_news.txt
│
├── config.py
├── kg_pipeline.py
├── README.md
```

### Description

| File             | Description                    |
| ---------------- | ------------------------------ |
| `kg_pipeline.py` | Main pipeline implementation   |
| `config.py`      | Database and API configuration |
| `data/`          | Financial news dataset         |
| `README.md`      | Project documentation          |

---

# Pipeline Description

## 1. Data Preprocessing

Raw financial news articles are cleaned before processing.

Operations include:

* removing HTML tags
* removing URLs
* removing special characters
* sentence tokenization
* filtering short sentences

Example preprocessing step:

```python
sentences = sent_tokenize(text)
```

Only meaningful sentences are passed to the extraction stage.

---

# Entity and Relationship Extraction

The system uses an **LLM with few-shot prompting** to extract structured triplets from financial text.

Triplets have the form:

```
Subject → Predicate → Object
```

Example:

```
Apple → ACQUIRES → TechCorp
Microsoft → REPORTS → quarterly earnings
Tesla → PARTNERS_WITH → Panasonic
```

Supported entity types:

* Company
* Person
* Product
* FinancialMetric
* Sector

Supported relationship types:

* ACQUIRES
* PARTNERS_WITH
* INVESTS_IN
* ANNOUNCES
* REPORTS
* INCREASES
* DECREASES
* LAUNCHES
* PROJECTS

---

# Data Cleaning

The extracted triplets may contain inconsistencies.
Therefore, a cleaning stage is applied.

Cleaning operations include:

* entity name normalization
* relationship validation
* removal of duplicate triplets
* validation of subject–predicate–object structure

Example normalization:

```
Apple Inc. → Apple
Microsoft Corporation → Microsoft
```

---

# Knowledge Graph Construction

The cleaned triplets are inserted into **Neo4j**.

Each entity becomes a **node**, and relationships become **edges**.

Example Cypher query:

```
MATCH (a:Company {name: "Apple"})
MATCH (b:Company {name: "TechCorp"})
MERGE (a)-[:ACQUIRES]->(b)
```

Nodes are automatically labeled based on their entity type.

Visualization colors:

| Entity Type     | Color  |
| --------------- | ------ |
| Company         | Orange |
| Person          | Blue   |
| Product         | Green  |
| FinancialMetric | Red    |
| Sector          | Purple |

---

# Natural Language Query Interface

Users can query the knowledge graph using natural language.

The system converts the question into a **Cypher query** using an LLM.

Example:

**User query**

```
Which companies has Apple acquired?
```

Generated Cypher:

```
MATCH (c1:Company {name: "Apple"})-[:ACQUIRES]->(c2:Company)
RETURN c1.name as Acquirer, c2.name as Acquired
```

Example result:

```
Apple → TechCorp
```

---

# Example Queries

You can try the following queries:

```
Which companies has Apple acquired?
What products has Tesla launched?
Show me all partnerships between companies
What financial metrics has Microsoft reported?
```

---

# Running the Project

## 1. Install dependencies

```bash
pip install py2neo nltk spacy pandas requests
```

Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## 2. Configure Neo4j

Edit `config.py`:

```
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
```

---

## 3. Run the pipeline

```
python kg_pipeline.py
```

The program will:

1. process financial news data
2. extract knowledge triplets
3. build the knowledge graph
4. start the query interface

---

# Results

The system successfully:

* processes financial news articles
* extracts entities and relationships
* builds a structured knowledge graph
* enables natural language querying of financial information

The resulting graph allows intuitive exploration of relationships between companies, products, and financial metrics.

---

# Conclusion

This project demonstrates how **LLM-based information extraction** can be combined with **graph databases** to construct a financial knowledge graph from unstructured text data.

The developed pipeline provides a scalable approach for transforming raw textual data into structured knowledge suitable for analytics, search, and decision support systems.

---
