"""
app.py — StructRAG Application Interface

Provides two modes:
  1. CLI (default):   Interactive query loop in the terminal.
  2. API (--api):     FastAPI server with POST /query endpoint.

Usage:
  python app.py             # Start interactive CLI
  python app.py --api       # Start FastAPI server on port 8000
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

from rag_engine import query_pipeline

# ── Logging Setup ─────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("structrag")


# ── CLI Mode ──────────────────────────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║              StructRAG — Vectorless RAG Engine            ║
║           Hierarchical JSON · Zero Vector DBs             ║
╚═══════════════════════════════════════════════════════════╝
  Type your question and press Enter.
  Type 'quit' or 'exit' to stop.
"""


def run_cli():
    """Interactive command-line query loop."""
    print(BANNER)

    while True:
        try:
            query = input("\n🔎 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        logger.info("Processing query: %s", query)

        try:
            result = query_pipeline(query)

            # ── Step 1 Log: Show which node was selected ──────────────────
            print(f"\n┌─ Step 1: Router ────────────────────────────────────")
            print(f"│  Selected node: {result['routed_node_id']}")
            print(f"│  Section title: {result['routed_title']}")
            print(f"└────────────────────────────────────────────────────\n")

            # ── Step 2 Output: Show the generated answer ──────────────────
            print(f"┌─ Step 2: Answer ────────────────────────────────────")
            for line in result["answer"].split("\n"):
                print(f"│  {line}")
            print(f"└────────────────────────────────────────────────────")

        except FileNotFoundError as e:
            logger.error("Knowledge tree not found: %s", e)
            print("💡 Run `python indexer.py docs/sample_policy.md` first to build the index.")
        except RuntimeError as e:
            logger.error("Pipeline error: %s", e)
        except Exception as e:
            logger.exception("Unexpected error: %s", e)


# ── API Mode (FastAPI) ────────────────────────────────────────────────────────

def create_api_app():
    """Create and return the FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(
        title="StructRAG API",
        description="Vectorless RAG using hierarchical JSON tree traversal and Groq.",
        version="1.0.0",
    )

    class QueryRequest(BaseModel):
        question: str

    class QueryResponse(BaseModel):
        routed_node_id: str
        routed_title: str
        answer: str

    @app.get("/")
    async def root():
        return {
            "service": "StructRAG",
            "status": "running",
            "usage": "POST /query with JSON body: {\"question\": \"your question\"}",
        }

    @app.post("/query", response_model=QueryResponse)
    async def handle_query(req: QueryRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        logger.info("API query received: %s", req.question)

        try:
            result = query_pipeline(req.question)

            logger.info(
                "Router selected node '%s' (%s)",
                result["routed_title"],
                result["routed_node_id"],
            )

            return QueryResponse(**result)

        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Knowledge tree not built. Run indexer.py first.",
            )
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

    return app


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server via Uvicorn."""
    import uvicorn

    logger.info("Starting StructRAG API on %s:%d", host, port)
    app = create_api_app()
    uvicorn.run(app, host=host, port=port)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="StructRAG — Vectorless RAG Engine",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start as a FastAPI server instead of interactive CLI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000).",
    )

    args = parser.parse_args()

    if args.api:
        run_api(port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
