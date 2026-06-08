"""
Task 4 — Chunking & Indexing vào Weaviate (local Docker).

Stack:
- Chunking: RecursiveCharacterTextSplitter (size=800, overlap=100)
  - 800 chars đủ context cho câu hỏi pháp lý dài; overlap 100 (~12%) tránh
    mất thông tin ở ranh giới chunk.
- Embedding: Cohere embed-v4.0 (1536 dim)
  - Multilingual, tốt cho tiếng Việt, state-of-the-art cho retrieval.
  - input_type="search_document" khi index, "search_query" khi search.
- Vector Store: Weaviate local Docker
  - Hỗ trợ hybrid search (dense + BM25) built-in — dùng trong Task 6 & 9.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# RecursiveCharacterTextSplitter: an toàn, hoạt động tốt với mọi loại văn bản
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# Cohere embed-v4.0: 1536 dim, multilingual, state-of-the-art cho tiếng Việt
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIM = 1536

VECTOR_STORE = "weaviate"
COLLECTION_NAME = "DrugLawDocs"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents dùng RecursiveCharacterTextSplitter.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            chunks.append({
                "content": text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks bằng Cohere embed-v4.0.
    input_type="search_document" cho document indexing.
    Tự động retry với exponential backoff khi bị rate-limit (429).
    """
    import time
    import cohere
    from cohere.errors import TooManyRequestsError

    co = cohere.ClientV2(COHERE_API_KEY)
    batch_size = 48  # Smaller batch to stay under 100k token/min limit
    n_batches = -(-len(chunks) // batch_size)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["content"] for c in batch]
        batch_num = i // batch_size + 1

        # Retry up to 5 times with exponential backoff
        for attempt in range(5):
            try:
                resp = co.embed(
                    model=EMBEDDING_MODEL,
                    input_type="search_document",
                    texts=texts,
                    embedding_types=["float"],
                )
                for chunk, emb in zip(batch, resp.embeddings.float):
                    chunk["embedding"] = list(emb)
                print(f"  Embedded batch {batch_num}/{n_batches}")
                break
            except TooManyRequestsError:
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s, ...
                print(f"  Rate limit on batch {batch_num}, waiting {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch {batch_num} after 5 retries")

    return chunks


def get_weaviate_client():
    import weaviate
    return weaviate.Client(WEAVIATE_URL)


def setup_schema(client):
    """Tạo Weaviate schema nếu chưa có."""
    existing = [c["class"] for c in client.schema.get().get("classes", [])]
    if COLLECTION_NAME in existing:
        return

    schema = {
        "class": COLLECTION_NAME,
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "doc_type", "dataType": ["text"]},
            {"name": "chunk_index", "dataType": ["int"]},
        ],
    }
    client.schema.create_class(schema)
    print(f"  Created schema: {COLLECTION_NAME}")


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào Weaviate."""
    client = get_weaviate_client()
    setup_schema(client)

    # Xoá data cũ
    try:
        client.schema.delete_class(COLLECTION_NAME)
        setup_schema(client)
    except Exception:
        pass

    with client.batch as batch:
        batch.batch_size = 100
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            props = {
                "content": chunk["content"],
                "source": meta.get("source", ""),
                "doc_type": meta.get("type", ""),
                "chunk_index": meta.get("chunk_index", 0),
            }
            batch.add_data_object(
                data_object=props,
                class_name=COLLECTION_NAME,
                vector=chunk["embedding"],
            )

    count = client.query.aggregate(COLLECTION_NAME).with_meta_count().do()
    total = count["data"]["Aggregate"][COLLECTION_NAME][0]["meta"]["count"]
    print(f"  Indexed {total} chunks in Weaviate")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} @ {WEAVIATE_URL}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Done.")


if __name__ == "__main__":
    run_pipeline()