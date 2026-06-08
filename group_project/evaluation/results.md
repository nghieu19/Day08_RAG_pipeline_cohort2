# RAG Evaluation Results

## Framework

Framework used: **Lightweight Local Evaluation**.

This evaluator is deterministic and runs offline. It approximates the required RAG metrics with token-overlap heuristics, so the scores are useful for comparing configs but should not be treated as an LLM-judge benchmark.

Note: `golden_dataset.json` currently has **3 Q&A pairs**, below the assignment requirement of 15+ pairs.

## Overall Scores

| Metric | Config A: hybrid + rerank | Config B: dense-only | Delta |
|--------|---------------------------|----------------------|-------|
| Faithfulness | 0.975 | 0.925 | +0.050 |
| Answer Relevance | 0.550 | 0.559 | -0.009 |
| Context Recall | 0.785 | 0.703 | +0.082 |
| Context Precision | 0.874 | 0.790 | +0.084 |
| **Average** | **0.796** | **0.744** | **+0.052** |

## A/B Comparison

- **Config A:** Task 9 hybrid retrieval: semantic search + lexical BM25, RRF merge, reranking, and PageIndex fallback.
- **Config B:** dense-only semantic search over the local vector index, without BM25 or reranking.

## Per-Question Results

| # | Question | Avg | Faithfulness | Relevance | Recall | Precision | Top Sources |
|---|----------|-----|--------------|-----------|--------|-----------|-------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự? | 0.792 | 1.000 | 0.459 | 0.710 | 1.000 | cac-toi-ve-ma-tuy-phan-tich-va-quy-dinh-blhs-2015.md, cac-toi-ve-ma-tuy-phan-tich-va-quy-dinh-blhs-2015.md, cac-toi-ve-ma-tuy-phan-tich-va-quy-dinh-blhs-2015.md |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | 0.949 | 1.000 | 0.857 | 1.000 | 0.940 | luat-phong-chong-ma-tuy-2021.md, luat-phong-chong-ma-tuy-2021.md, luat-phong-chong-ma-tuy-2021.md |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào? | 0.646 | 0.924 | 0.333 | 0.645 | 0.683 | Nghị định 28_2026_NĐ-CP_ Quy định danh mục chất ma túy và tiền chất mới nhất.md, cac-toi-ve-ma-tuy-phan-tich-va-quy-dinh-blhs-2015.md, nghi-dinh-105-2021.md |

## Worst Performers

| # | Question | Average | Likely Failure Stage | Root Cause |
|---|----------|---------|----------------------|------------|
| 1 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào? | 0.646 | Generation | Relevant context exists, but the extractive answer does not fully match the expected answer. |
| 2 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự? | 0.792 | Generation | Relevant context exists, but the extractive answer does not fully match the expected answer. |
| 3 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | 0.949 | Generation | Relevant context exists, but the extractive answer does not fully match the expected answer. |

## Recommendations

1. Expand `golden_dataset.json` to at least 15 Q&A pairs covering legal penalties, definitions, drug schedules, news, and follow-up questions.
2. Clean table-heavy legal documents, especially drug schedule decrees, so chunking preserves rows and headings.
3. Add specialized answer templates for common legal intents: definition, penalty range, rehabilitation forms, and controlled-substance lists.
4. When API quota is available, add DeepEval or RAGAS as an optional LLM-judge layer on top of this local baseline.
