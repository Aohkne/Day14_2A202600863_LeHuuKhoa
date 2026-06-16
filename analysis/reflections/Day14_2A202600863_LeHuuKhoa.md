# Reflection — Lê Hữu Khoa (2A202600863)

## Lab Day 14: AI Evaluation Factory

---

## 1. Đóng góp cá nhân

| Module | Công việc cụ thể |
|--------|-----------------|
| **Agent (`agent/main_agent.py`)** | Xây dựng `MainAgent` với BM25-style keyword retrieval (CHUNK_SIZE=800, OVERLAP=100), diversity cap 2 chunks/doc, trả về `retrieved_ids` trong metadata để RAG evaluator tính Hit Rate thực tế. |
| **Runner (`engine/runner.py`)** | Triển khai `BenchmarkRunner` với async batch runner: chạy N cases song song trong từng batch, giới hạn batch_size để không trigger rate limit. |
| **RAG Evaluator (`main.py`)** | Thiết kế `RAGEvaluator` tích hợp `RetrievalEvaluator` thực tế khi agent trả về `retrieved_ids`; fallback sang difficulty-based simulation khi không có metadata. |
| **API Integration** | Tích hợp Fireworks AI (OpenAI-compatible endpoint) với `base_url` config, kiểm tra model availability và đảm bảo `response_format={"type":"json_object"}` cho judge responses đáng tin cậy. |
| **Reflection & Analysis** | Phân tích kỹ thuật các vấn đề về retrieval diversity, hallucination trap, và prompt injection từ kết quả benchmark thực tế. |

---

## 2. Kiến thức kỹ thuật đã học được

### MRR (Mean Reciprocal Rank)

MRR = trung bình của `1/rank_i`, trong đó `rank_i` là vị trí 1-indexed của document đúng đầu tiên trong danh sách retrieved.

Ví dụ thực tế từ lab này:
- Case `momo_terms_of_service`: retrieved = `[momo_terms_of_service, shopee_terms_of_service, ...]` → rank=1 → MRR contribution = 1.0
- Case `zalopay_terms_of_service`: retrieved = `[momo_terms_of_service, shopee_terms, zalopay_terms_of_service, ...]` → rank=3 → MRR contribution = 0.33

**Trade-off Hit Rate vs MRR:** Hit Rate@5 = 1.0 (tìm thấy trong top-5) nhưng MRR = 0.33 (tìm thấy nhưng ở vị trí thấp). Hệ thống có Hit Rate cao nhưng MRR thấp nghĩa là retriever tìm đúng nhưng xếp hạng sai — đây là dấu hiệu cần cải thiện reranker.

### Cohen's Kappa vs Agreement Rate đơn giản

Agreement Rate đơn giản = `số case đồng ý / tổng case`. Điểm yếu: nếu 2 judges đều có xu hướng cho điểm cao, agreement vô tình cao dù không có ý nghĩa thực sự.

Cohen's Kappa = `(P_observed - P_expected) / (1 - P_expected)` — trừ đi xác suất đồng ý ngẫu nhiên.

Trong project này, tôi dùng Agreement Rate với `delta ≤ 1` làm threshold vì:
- Rubric 1-5 của chúng ta là định tính, không phải đo lường tuyệt đối
- Hai models (`deepseek-v4-pro` và `kimi-k2p5`) có calibration khác nhau → delta ≤ 1 là acceptable

### Position Bias trong LLM Judge

LLM judge có xu hướng thiên vị response được đặt ở vị trí đầu tiên (Response A). `check_position_bias()` hoán đổi thứ tự A/B hai lần:
- Nếu judge luôn chọn "A" bất kể nội dung → position bias = True
- Nếu judge chọn nhất quán theo nội dung dù đổi vị trí → bias = False

Cách giảm position bias: Tournament-style ranking (so sánh nhiều cặp), hoặc lấy average score thay vì so sánh đôi.

### Trade-off Chi phí vs Chất lượng Eval

Từ kết quả benchmark thực tế (85 cases, 2 judges, 2 versions = 340 judge calls):
- `deepseek-v4-pro`: $3.00/1M input — chất lượng cao, phân tích sâu
- `kimi-k2p5`: $0.90/1M input — nhanh hơn, đủ tốt cho rubric định tính

**3 cách giảm 30% chi phí mà không giảm độ chính xác:**
1. **Caching judge results:** Cache theo (question_hash, answer_hash) — V1 và V2 cùng câu hỏi chỉ khác answer → chỉ re-judge answer mới
2. **Tiered judging:** Chỉ dùng Judge B (`kimi-k2p5`, rẻ hơn) cho tất cả cases, escalate lên Judge A (`deepseek-v4-pro`) khi score trong vùng borderline (2-3)
3. **Reduce max_tokens:** Giảm từ 150 xuống 80 tokens cho judge (chỉ cần score + 1 câu reason) — tiết kiệm ~30% output tokens

---

## 3. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: Fireworks API — Model không tìm thấy

**Vấn đề:** Code ban đầu dùng `gpt-4o-mini` / `gpt-3.5-turbo` nhưng `.env` chứa Fireworks API key (`fw_...`). Gọi API trả về `404 Model not found`.

**Root cause:** Fireworks AI có OpenAI-compatible endpoint nhưng model names khác hoàn toàn. Hơn nữa, code thiếu `base_url=os.getenv("FIREWORKS_BASE_URL")` trong `AsyncOpenAI()`.

**Giải quyết:**
1. Thêm `base_url` vào cả `AsyncOpenAI` trong `agent/main_agent.py` và `engine/llm_judge.py`
2. List models qua API để xác định models khả dụng: `deepseek-v4-pro`, `kimi-k2p5`, `kimi-k2p6`, `glm-5p1`
3. Test từng model với judge prompt → chọn `deepseek-v4-pro` (JUDGE_A) và `kimi-k2p5` (JUDGE_B) vì cả hai support `response_format={"type":"json_object"}`

### Vấn đề 2: Judge LLM trả về JSON lẫn với prose

**Vấn đề:** Một số models (glm-5p1, kimi-k2p6) thêm text giải thích trước JSON dù prompt nói "ONLY JSON". `json.loads()` raise `JSONDecodeError`.

**Giải quyết:**
- Thêm `response_format={"type": "json_object"}` — buộc các model hỗ trợ phải output pure JSON
- Thêm regex fallback: `re.search(r'\{[^{}]+\}', content, re.DOTALL)` để extract JSON từ mixed output
- Chọn models có JSON mode support thay vì models không đáng tin cậy

### Vấn đề 3: Cross-document retrieval thất bại nhất quán

**Vấn đề:** Cases so sánh cross-document (e.g., MoMo vs ZaloPay) luôn có Hit Rate thấp. BM25 retriever bị dominated bởi document dài hơn vì có nhiều chunks hơn → score tổng cao hơn dù không relevant với sub-question kia.

**Giải quyết (ngắn hạn):** `RAGEvaluator` fallback simulation dùng difficulty-based hit rate (cross-document = 0.40) để phản ánh thực tế.

**Giải quyết (dài hạn):** Cần Query Decomposition — tách câu hỏi so sánh thành N sub-queries, retrieve riêng cho từng sub-query, merge kết quả và đảm bảo mỗi required document đều được đại diện.

---

## 4. Nhìn lại và đề xuất cải tiến

**Điều tôi làm tốt:**
- Xây dựng BM25-style retriever với document diversity cap (max 2 chunks/doc) — đây là improvement quan trọng so với naive top-K
- Async batch runner với configurable batch_size giúp cân bằng tốc độ và rate limit
- Phát hiện và xử lý vấn đề API compatibility (Fireworks + JSON mode) trong thực chiến

**Điều tôi sẽ làm khác nếu có thêm thời gian:**

1. **Semantic Chunking:** Dùng `langchain.text_splitter.RecursiveCharacterTextSplitter` với separators=["\n## ", "\n### ", "\n\n", "\n"] để giữ nguyên bảng biểu và tiêu đề section trong cùng một chunk

2. **Vector Retrieval thực sự:** Tích hợp ChromaDB/Qdrant với sentence-transformers (e.g., `paraphrase-multilingual-mpnet-base-v2`) thay vì BM25 keyword matching — đặc biệt quan trọng với policy documents tiếng Việt

3. **Query Decomposition:** Trước khi retrieve, dùng LLM để tách câu hỏi phức tạp thành sub-queries đơn giản hơn

4. **Calibration với Human Labels:** Thu thập ground truth scores từ người (5 câu hỏi × 3 người chấm) để tính Inter-Annotator Agreement, sau đó so sánh với LLM judge agreement → validate độ tin cậy của automated scoring system

---

*Thực hiện: Lê Hữu Khoa — 2A202600863*
*Ngày: 2026-06-16*
