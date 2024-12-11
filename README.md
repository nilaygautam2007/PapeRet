
# PapeRet: Research Papers Retrieval System

## Overview
This project implements a comprehensive Information Retrieval (IR) system designed to retrieve research papers based on metadata, abstracts, and full-text data. The system includes robust indexing, query processing, ranking, and relevance evaluation features, integrated with advanced models like LLAMA for query augmentation and summarization. A user-friendly Streamlit app (`PapeRet.py`) enables interactive search and retrieval.

---

## Features

- **Streamlit Interface**:
  - Search for research papers interactively.
  - Optional LLAMA-enhanced query processing.
  - AI-generated summaries of retrieved documents.
- **Indexing**:
  - `BasicInvertedIndex` for efficient term-document mappings.
  - Dynamic computation of document statistics like term frequencies and document frequencies.
- **Query Processing**:
  - Tokenization and stopword removal.
  - Augmented query processing with title index integration.
  - Enhanced query understanding with author and year filters.
- **Ranking and Scoring**:
  - BM25 and TF-IDF relevance scoring algorithms.
  - LLAMA-powered query-to-keyword extraction for improved ranking.
  - Dynamic re-ranking based on query-matched authors and publication years.
- **Relevance Evaluation**:
  - Metrics such as MAP (Mean Average Precision) and NDCG (Normalized Discounted Cumulative Gain).
  - Pre-curated relevance scores for benchmarking rankers.
- **Document Preprocessing**:
  - Regex tokenization.

---

## Directory Structure

```
IR_Project/
├── all_papers_index/   
│   ├── index.json            # Main paper text index
├── all_papers_title_index/   
│   ├── index.json            # Title index
├── all_authors.pkl           # List of all authors
├── docid_authors_map.pkl     # Mapping of document IDs to authors
├── docid_link_map.pkl        # Mapping of document IDs to urls
├── docid_title_map.pkl       # Mapping of document IDs to titles
├── docid_year_map.pkl        # Mapping of document IDs to years
├── docid_abstract_map.pkl    # Mapping of document IDs to abstracts
├── docid_year_map.pkl        # Mapping of document IDs to publication years
├── stopwords.txt             # List of stopwords for filtering
├── PapeRet.py                # Streamlit app
├── main.py                   # System initialization
├── indexing.py               # Indexing strategies
├── document_preprocessor.py  # Tokenization and preprocessing
├── custom_ranker.py          # Ranking and scoring algorithms
├── relevance.py              # Relevance evaluation and testing
├── llama_tokenise_rag.py     # LLAMA integration for summarization and query extraction
├── relevance.py              # Rough notebook used to create miscellaneous files and tests 
├── ArXiv/                    # Scripts for obtaining paper corpus and text from ArXiv.
├── OpenAlex/                 # Scripts for fetching metadata of seed papers, their references, and scraping text from 15 websites.
└── README.md                 # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IR_Project.git
   cd IR_Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up necessary data files in the directory:
   - Place `.pkl`, `index` files for mappings.
   - Add `stopwords.txt` for stopword filtering.
   - Files can be found [here.](https://drive.google.com/drive/folders/131ffNJDOY0wx-YzeaOj2EZIw8HPP9nkR?usp=share_link)

4. Add login key for huggingface_hub for in setup_pipeline function in llama_tokenise_rag.py 

---

## Usage

### Running the Streamlit App

To start the application, run the following command:
```bash
streamlit run PapeRet.py
```
The app interface allows you to:
- Input search queries and retrieve results.
- Toggle LLAMA integration for enhanced query processing and AI-generated summaries.

---

## Results
The system achieves high precision and recall across multiple datasets. Evaluation results include:
- Mean Average Precision (MAP) and NDCG metrics.
- CSV outputs for analysis (`results/final_results.csv`).

---

## Future Work
- Integrate deep learning-based rankers for improved scoring.
- Enhance real-time query recommendations.
- Extend support for multilingual datasets.

---

## Contributors
- **Nilay Gautam**
- **Rishikesh Ksheersagar**

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
