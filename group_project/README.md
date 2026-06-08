# Bài Tập Nhóm - RAG Chatbot

## Mục Tiêu

Xây dựng chatbot trả lời câu hỏi về pháp luật ma túy và các tin tức liên quan, sử dụng pipeline cá nhân đã hoàn thành:

```text
Streamlit Chat UI
    -> Task 9 Retrieval Pipeline
        -> Semantic Search + Lexical BM25
        -> RRF Merge + Reranking
        -> PageIndex-style fallback
    -> Task 10 Generation With Citation
    -> Source Display + Conversation Memory
```

## Yêu Cầu 1: Sản Phẩm Nhóm RAG Chatbot

Đã triển khai tại:

```text
group_project/app.py
```

Tính năng:

- Giao diện chat bằng Streamlit.
- Trả lời có citation dựa trên `src/task10_generation.py`.
- Hỗ trợ follow-up questions bằng conversation memory trong `st.session_state`.
- Hiển thị source documents đã dùng cho từng câu trả lời.
- Cho phép chỉnh `top_k` và xem formatted context khi cần debug.

## Kiến Trúc Hệ Thống

```text
User
  |
  v
Streamlit Chatbot (group_project/app.py)
  |
  v
generate_with_citation() - Task 10
  |
  v
retrieve() - Task 9
  |
  +--> semantic_search() - Task 5
  +--> lexical_search()  - Task 6
  +--> rerank()          - Task 7
  +--> pageindex_search() fallback - Task 8
  |
  v
Answer + citations + source chunks
```

## Hướng Dẫn Chạy

Chạy từ thư mục root của repo:

```bash
pip install -r requirements.txt
streamlit run group_project/app.py
```

Để bật câu trả lời kiểu ChatGPT qua OpenAI API, tạo file `.env` ở root repo:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Nếu index chưa có, chạy lại Task 4 trước:

```bash
python src/task4_chunking_indexing.py
```

## Câu Hỏi Demo Gợi Ý

```text
Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?
```

```text
Những nghệ sĩ nào trong dữ liệu tin tức có liên quan tới ma túy?
```

```text
Luật phòng chống ma túy nói gì về cai nghiện?
```

Follow-up:

```text
Nguồn nào nói điều đó?
```

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Thành viên 1 | TBD | Data collection + conversion | Done |
| Thành viên 2 | TBD | Retrieval pipeline + reranking | Done |
| Thành viên 3 | TBD | Chatbot UI + citation display | Done |
| Thành viên 4 | TBD | Evaluation pipeline + report | Pending |

## Ghi Chú

App hiện chạy offline bằng pipeline local trong `src/`, không yêu cầu OpenAI API key hoặc PageIndex API key. Khi nhóm có API key thật, có thể thay phần generation hoặc PageIndex fallback bằng dịch vụ tương ứng mà không đổi giao diện.
