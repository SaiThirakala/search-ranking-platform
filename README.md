# Intelligent Search + Ranking Platform  
### Part 1: BM25 Research Paper Search Engine

An end-to-end search platform built on a real-world research paper corpus.  
This Part 1 implements a **containerized FastAPI backend** and **React frontend** that allow users to search research papers using **BM25 keyword ranking** over titles and abstracts.

The project uses the **DBLP v10 dataset** (~800k research paper records) and demonstrates core machine learning engineering concepts including:

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