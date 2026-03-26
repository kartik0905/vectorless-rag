"""
indexer.py — Document Ingestion & Knowledge Tree Builder

Parses a Markdown file by headers (#, ##, ###), generates LLM summaries
for each section via Groq, and outputs a structured knowledge_tree.json.
"""

import json
import re
import sys
import os
import time
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"  # Cheap, fast model for summarization
OUTPUT_FILE = "knowledge_tree.json"
MAX_RETRIES = 5                      # Max retries on rate-limit errors
BASE_BACKOFF = 2                     # Base seconds for exponential backoff


def init_groq_client() -> Groq:
    """Initialize the Groq client, validating the API key is present."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("❌ Error: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    return Groq(api_key=api_key)


# ── Markdown Parsing ─────────────────────────────────────────────────────────

def parse_markdown(filepath: str) -> list[dict]:
    """
    Parse a Markdown file into chunks split on header lines.

    Returns a list of dicts:
        { "title": "Header Title", "level": 1-3, "content": "raw text..." }
    """
    path = Path(filepath)
    if not path.exists():
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    # Regex matches lines starting with 1-3 '#' followed by a space
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    chunks = []
    matches = list(header_pattern.finditer(text))

    if not matches:
        print("⚠️  Warning: No Markdown headers found. The entire file will be one node.")
        chunks.append({
            "title": path.stem,
            "level": 1,
            "content": text.strip(),
        })
        return chunks

    for i, match in enumerate(matches):
        level = len(match.group(1))      # Number of '#' characters
        title = match.group(2).strip()

        # Content = text between this header and the next (or end of file)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        chunks.append({
            "title": title,
            "level": level,
            "content": content,
        })

    print(f"📄 Parsed {len(chunks)} sections from '{filepath}'")
    return chunks


# ── LLM Summarization ────────────────────────────────────────────────────────

def generate_summary(client: Groq, title: str, content: str) -> str:
    """
    Call Groq to produce a 1-2 sentence summary of a document section.
    Includes exponential backoff for rate-limit handling.
    """
    if not content:
        return f"Section '{title}' has no body text."

    # Truncate very long sections to keep token usage low
    truncated = content[:3000] if len(content) > 3000 else content

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a technical writer. Summarize the following document "
                            "section in exactly 1-2 concise sentences. Return ONLY the summary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Section Title: {title}\n\n{truncated}",
                    },
                ],
                temperature=0.2,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            wait = BASE_BACKOFF ** (attempt + 1)
            print(f"   ⏳ Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)

    # Fallback if all retries exhausted
    print(f"   ⚠️  Could not summarize '{title}' after {MAX_RETRIES} retries. Using fallback.")
    return f"Section covering: {title}"


def generate_node_id(title: str, index: int) -> str:
    """Generate a unique, deterministic node ID from the title and index."""
    raw = f"{index}:{title}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


# ── Tree Construction ─────────────────────────────────────────────────────────

def build_knowledge_tree(filepath: str) -> dict:
    """
    Full pipeline: parse Markdown → summarize each section → build JSON tree.
    """
    client = init_groq_client()
    chunks = parse_markdown(filepath)

    # Extract document title from the first H1, or use filename
    doc_title = next(
        (c["title"] for c in chunks if c["level"] == 1),
        Path(filepath).stem,
    )

    nodes = []
    for i, chunk in enumerate(chunks):
        node_id = generate_node_id(chunk["title"], i)
        print(f"   🔍 [{i+1}/{len(chunks)}] Summarizing: {chunk['title']}")

        summary = generate_summary(client, chunk["title"], chunk["content"])

        nodes.append({
            "node_id": node_id,
            "title": chunk["title"],
            "summary": summary,
            "content": chunk["content"],
        })

        # Small delay between calls to stay within Groq free-tier limits
        time.sleep(0.5)

    tree = {
        "document": doc_title,
        "nodes": nodes,
    }
    return tree


def save_tree(tree: dict, output_path: str = OUTPUT_FILE) -> None:
    """Write the knowledge tree to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Knowledge tree saved to '{output_path}' ({len(tree['nodes'])} nodes)")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <path-to-markdown-file>")
        print("Example: python indexer.py docs/sample_policy.md")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"  StructRAG Indexer — Building Knowledge Tree")
    print(f"{'='*60}\n")

    tree = build_knowledge_tree(filepath)
    save_tree(tree)


if __name__ == "__main__":
    main()
