# Intelligent Search + Ranking Platform  
### Part 1: BM25 Research Paper Search Engine

An end-to-end search platform built on a real-world research paper corpus.  
This Part 1 implements a **containerized FastAPI backend** and **React frontend** that allow users to search research papers using **BM25 keyword ranking** over titles and abstracts.

The project uses the **DBLP v10 dataset** and demonstrates core machine learning engineering concepts including:

- Data preprocessing pipelines
- Information retrieval systems
- API development
- Frontend / backend integration
- Dockerized local development
- Search relevance ranking

---

# Demo Overview

Users can:
- Search for topics by keywords

The system returns ranked papers containing :
- Title
- Author
- Venue
- Publication Year
- Citation count
- Abstract
- BM25 score

---

# Teck Stack

## Backend
- Python 3.11
- FastAPI
- Uvicorn
- Pandas
- rank_bm25
- Pydantic

## Frontend
- React
- Vite
- CSS

## Infrastructure
- Docker

## Dataset
- DBLP v10 research paper metadata dataset (https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset/data)
- Roughly 1,000,000 unique records
- Data has fields 
    - id: str
    - title: str
    - authors: str
    - venue: str
    - year: int
    - n_citations: int
    - abstract: str
- For search, each paper is indexed as title + abstract

--- 

## Preprocessing Pipeline

1. Read raw CSV file
2. Select only the required columns
3. Clean whitespace and missing values
4. Filter out "bad" rows
5. Build searchable text (title + abstract)
6. Saved processed records to data/processed/papers_processed.jsonl

---

## Ranking Model: BM25

