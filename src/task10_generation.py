"""
Task 10 — Generation Có Citation.

- top_k=5: đủ context, không gây lost-in-the-middle với prompt dài
- top_p=0.9: nucleus sampling, giữ đa dạng nhưng không quá random
- temperature=0.3: factual, ít hallucinate — phù hợp cho RAG pháp lý
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle".

    LLM nhớ tốt thông tin ở đầu và cuối, quên thông tin ở giữa.
    Strategy: chunk quan trọng nhất (rank 1) → đầu, quan trọng thứ 2 → cuối,
    các chunk kém quan trọng hơn ở giữa.

    Input (sorted by score desc): [1, 2, 3, 4, 5]
    Output:                        [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    # Tách: odd indices (0,2,4,...) đặt ở đầu theo thứ tự
    # even indices (1,3,5,...) đặt ở cuối theo thứ tự ngược
    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    back.reverse()
    return front + back


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string có source labels cho citation.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation với citation.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key == "sk-xxx":
        # Không có API key: trả về context trực tiếp kèm disclaimer
        answer = (
            "⚠ OpenAI API key chưa được cấu hình.\n\n"
            "Dựa trên context đã retrieve:\n\n" + context[:1500]
        )
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }

    from openai import OpenAI

    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer'][:500]}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")