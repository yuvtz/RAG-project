# Document Indexing & Semantic Search (Gemini + Postgres + pgvector)

This project implements a **lightweight RAG (Retrieval-Augmented Generation)** pipeline that indexes documents (`.pdf`, `.docx`) into a PostgreSQL database using **Google Gemini embeddings**, and performs **semantic search** using `pgvector`.

---

## Overview

![Architecture Diagram](https://github.com/yourusername/document-rag/assets/architecture_example.png)

A simple two-stage pipeline:

1. **Indexing:** Extract and embed text chunks into PostgreSQL (with `pgvector`).
2. **Searching:** Query semantically similar chunks using cosine similarity.

---

## Features

- Extract text from **PDF** and **DOCX**
- Clean and split documents logically (sentences / paragraphs / fixed size)
- Generate embeddings via **Gemini `text-embedding-004`**
- Store vectors in **Postgres + pgvector**
- Perform **semantic search**
- Command-line interface for ease of use

---

## Installation

### 1. Clone & setup environment
```bash
git clone https://github.com/yourusername/document-rag.git
cd document-rag
python -m venv venv
source venv/bin/activate   # or on Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
POSTGRES_URL=postgresql://username:password@localhost:5432/your_database
```

Ensure PostgreSQL is running and the `pgvector` extension is installed.

---

## Usage

### Index Documents
```bash
python index_documents.py --paths docs/ --strategy paragraphs
```

**Options:**

| Flag | Description | Example |
|------|--------------|---------|
| `--paths` | One or more files, directories, or glob patterns | `--paths docs/` |
| `--strategy` | Splitting strategy: `fixed`, `sentences`, `paragraphs` | `--strategy sentences` |
| `--chunk-size` | Chunk size (used for `fixed`) | `--chunk-size 800` |
| `--overlap` | Overlap between chunks (for `fixed`) | `--overlap 120` |
| `--model` | Gemini embedding model | `--model text-embedding-004` |

---

### Search Documents
```bash
python search_documents.py -q "cleaning lyrics dataset" -k 5
```

**Options:**

| Flag | Description | Default |
|------|--------------|----------|
| `-q` | Search text | *(required)* |
| `-k` | Number of results | `5` |
| `--model` | Embedding model | `text-embedding-004` |

**Example output:**

```
Top 5 results for: cleaning lyrics dataset
------------------------------------------------------------
1. [0.7779] Report.docx
   The result is cleaned lyrics in both train_data and test_data...
```

---

## Database Schema

```sql
CREATE TABLE public.document_chunks (
    id             SERIAL PRIMARY KEY,
    chunk_text     TEXT NOT NULL,
    embedding      VECTOR(768) NOT NULL,
    filename       TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## Example Workflow

1. Add your files to `docs/`.
2. Index them:
   ```bash
   python index_documents.py --paths docs/ --strategy paragraphs
   ```
3. Search semantically:
   ```bash
   python search_documents.py -q "reset procedure for device"
   ```

---

## Troubleshooting

| Issue | Cause | Fix |
|--------|--------|------|
| `RuntimeError: GEMINI_API_KEY missing` | Missing `.env` file | Add `.env` to project root |
| `Unexpected embedding response` | Gemini response shape | Update: `pip install -U google-generativeai` |
| `psycopg2.OperationalError` | Invalid DB URL or Postgres down | Check `POSTGRES_URL` |
| Irrelevant results | No semantic match | Try different query or re-index |

---

## License

MIT License © 2025 Yuval Tzur
