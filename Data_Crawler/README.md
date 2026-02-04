# Data Crawler & Data Cleaner (Multi-Agent)

Course work for **Big Data Analysis (BIT, 2025–2026)**.  
This project implements a simple **multi-agent pipeline** to crawl data from a target source, clean/normalize it, and export results as JSON for further analysis.

## Features
- Web data crawling (collects raw items)
- Data cleaning (normalization, filtering, deduplication where applicable)
- JSON export: raw / cleaned / evaluation (if enabled)

## Tech Stack
Python · Requests/HTTP · Parsing · Data Cleaning · JSON · (Multi-Agent logic)

## Project Structure
```text
Data_Crawler/
  main.py
  crawler/
    intelligent_crawler.py
    data_cleaner.py
  outputs/
  requirements.txt
