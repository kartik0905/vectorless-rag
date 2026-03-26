# StructRAG — Vectorless RAG Engine

A lightweight, high-performance Retrieval-Augmented Generation system that uses **hierarchical JSON tree traversal** instead of vector databases. Built to stay within Groq's free-tier limits.

## How It Works

```
┌───────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Markdown Doc │────▶│    indexer.py     │────▶│ knowledge_tree.json│
│  (Your Data)  │     │ Parse + Summarize │     │  Structured Tree   │
└───────────────┘     └──────────────────┘     └────────┬──────────┘
                                                        │
                      ┌─────────────────────────────────┘
                      ▼
               ┌─────────────┐    Step 1: Router     ┌──────────────┐
               │ User Query  │───────────────────────▶│ Select Node  │
               │             │   (ToC only, low       │ (node_id)    │
               │             │    tokens)             └──────┬───────┘
               │             │                               │
               │             │    Step 2: Generator   ┌──────▼───────┐
               │             │───────────────────────▶│ Full Answer  │
               └─────────────┘   (full section text)  └──────────────┘
```

**Zero vectors. Zero embeddings. Just structured JSON + smart prompting.**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add your Groq API key from https://console.groq.com

# 3. Index a document
python indexer.py docs/sample_policy.md

# 4. Ask questions
python app.py
```

## API Mode

```bash
# Start the FastAPI server
python app.py --api --port 8000

# Query via curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?"}'
```

## Project Structure

```
├── indexer.py              # Markdown parser + Groq summarizer
├── rag_engine.py           # 2-call retrieval pipeline
├── app.py                  # CLI + FastAPI interface
├── docs/
│   └── sample_policy.md    # Sample document for testing
├── requirements.txt
├── .env.example
└── knowledge_tree.json     # Generated after indexing
```

## Models Used

| Step | Model | Purpose |
|------|-------|---------|
| Indexing | `llama-3.1-8b-instant` | Generate section summaries |
| Routing | `llama-3.1-8b-instant` | Select relevant section (low tokens) |
| Generation | `llama-3.3-70b-versatile` | Produce detailed answers |
