"""
rag_engine.py — 2-Call Retrieval & Generation Pipeline

Step 1 (Router):  Send a lightweight "Table of Contents" to an LLM.
                  The LLM returns the single best-matching node_id.

Step 2 (Generator): Fetch the full content for that node and generate
                     a grounded answer using a stronger LLM.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

ROUTER_MODEL = "llama-3.1-8b-instant"    # Fast, cheap model for routing
GENERATOR_MODEL = "llama-3.3-70b-versatile"  # Stronger model for answer generation
KNOWLEDGE_FILE = "knowledge_tree.json"
MAX_RETRIES = 5
BASE_BACKOFF = 2


def init_groq_client() -> Groq:
    """Initialize the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)


# ── Knowledge Tree Loader ────────────────────────────────────────────────────

def load_knowledge_tree(path: str = KNOWLEDGE_FILE) -> dict:
    """Load the knowledge tree JSON from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Knowledge tree not found at '{path}'. Run indexer.py first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table_of_contents(tree: dict) -> str:
    """
    Extract a lightweight Table of Contents from the knowledge tree.
    Includes only node_id, title, and summary — NOT the full content.
    """
    lines = [f"Document: {tree['document']}\n"]
    for node in tree["nodes"]:
        lines.append(
            f"- node_id: {node['node_id']}  |  "
            f"title: {node['title']}  |  "
            f"summary: {node['summary']}"
        )
    return "\n".join(lines)


def find_node_by_id(tree: dict, node_id: str) -> dict | None:
    """Look up a node by its node_id. Returns None if not found."""
    for node in tree["nodes"]:
        if node["node_id"] == node_id:
            return node
    return None


# ── Step 1: Router ────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a router. Read the user's query and the provided table of contents.
Return ONLY the exact `node_id` of the single section that is most likely to contain the answer.
Do not return any other text, explanation, or formatting. Just the node_id string."""


def route_query(client: Groq, query: str, toc: str) -> str:
    """
    Send the query + Table of Contents to the router model.
    Returns the raw node_id string from the LLM.
    """
    user_message = (
        f"## User Query\n{query}\n\n"
        f"## Table of Contents\n{toc}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,   # Deterministic routing
                max_tokens=50,     # node_id is very short
            )
            raw = response.choices[0].message.content.strip()
            # Clean any accidental quotes or whitespace
            cleaned = raw.strip("`\"' \n")
            # Try to extract a hex-like node_id if the model added extra text
            match = re.search(r"[a-f0-9]{10}", cleaned)
            return match.group(0) if match else cleaned

        except RateLimitError:
            wait = BASE_BACKOFF ** (attempt + 1)
            print(f"   ⏳ Router rate limited. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Router failed after max retries due to rate limiting.")


# ── Step 2: Generator ─────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """You are a helpful assistant. Use ONLY the provided document section to answer the user's question.
If the answer is not in the provided text, say: "I could not find the answer in the provided document section."
Do not make up information or use external knowledge."""


def generate_answer(client: Groq, query: str, node: dict) -> str:
    """
    Send the query + the full node content to the generator model.
    Returns the final answer string.
    """
    user_message = (
        f"## User Question\n{query}\n\n"
        f"## Document Section: {node['title']}\n{node['content']}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            wait = BASE_BACKOFF ** (attempt + 1)
            print(f"   ⏳ Generator rate limited. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Generator failed after max retries due to rate limiting.")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def query_pipeline(
    query: str,
    knowledge_path: str = KNOWLEDGE_FILE,
) -> dict:
    """
    Execute the full 2-call RAG pipeline.

    Returns a dict with:
        - routed_node_id: the node selected by Step 1
        - routed_title:   the human-readable title of that node
        - answer:         the generated answer from Step 2
    """
    client = init_groq_client()
    tree = load_knowledge_tree(knowledge_path)
    toc = build_table_of_contents(tree)

    # ── Step 1: Route ─────────────────────────────────────────────────────
    routed_id = route_query(client, query, toc)
    node = find_node_by_id(tree, routed_id)

    # Handle hallucinated / invalid node_id
    if node is None:
        # Attempt fuzzy fallback: find the node whose title best matches
        # by checking if the routed text appears in any title
        print(f"   ⚠️  Router returned unknown node_id: '{routed_id}'")
        print("   🔄 Attempting title-based fallback match...")

        for candidate in tree["nodes"]:
            if routed_id.lower() in candidate["title"].lower():
                node = candidate
                routed_id = candidate["node_id"]
                print(f"   ✅ Fallback matched: '{node['title']}'")
                break

        if node is None:
            return {
                "routed_node_id": routed_id,
                "routed_title": "❌ NOT FOUND",
                "answer": (
                    f"The router returned node_id '{routed_id}' which does not exist "
                    f"in the knowledge tree. This may be an LLM hallucination. "
                    f"Please try rephrasing your question."
                ),
            }

    # ── Step 2: Generate ──────────────────────────────────────────────────
    answer = generate_answer(client, query, node)

    return {
        "routed_node_id": routed_id,
        "routed_title": node["title"],
        "answer": answer,
    }
