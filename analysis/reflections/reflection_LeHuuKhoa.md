# Reflection — Lê Hữu Khoa (2A202600863)

## Lab Day 14: AI Evaluation Factory

---

## 1. Đóng góp cá nhân

| Module | Công việc cụ thể |
|--------|-----------------|
| **Agent (`agent/main_agent.py`)** | Xây dựng `MainAgent` với BM25-style keyword retrieval (CHUNK_SIZE=800, OVERLAP=100); thêm tham số `version` để V1/V2 thực sự khác nhau về hành vi (V1 = top-k thuần, V2 = top-k + document-diversity cap 2 chunk/doc), thay vì chạy cùng 1 agent 2 lần như bản gốc — cần thiết để Regression Testing có ý nghĩa thật. |
| **Multi-Judge (`engine/llm_judge.py`)** | Chuyển từ Fireworks sang OpenAI trực tiếp (`gpt-4o` + `gpt-4o-mini`); triển khai `get_cohens_kappa()` tính Cohen's Kappa thật (không chỉ agreement_rate ngây thơ) và sửa `get_cost_report()` để tính giá theo đúng model của từng judge (code cũ áp giá của Judge A cho toàn bộ token, gây sai lệch chi phí). |
| **Runner (`engine/runner.py`)** | Tăng `batch_size` từ 5 → 15 sau khi đo được pipeline gốc chạy 85 case mất ~4 phút/version — vượt ngưỡng "<2 phút/50 case" của rubric. Sau khi tăng concurrency: 170 lượt đánh giá (V1+V2) chỉ còn 72 giây. |
| **RAG Evaluator (`main.py`)** | `RAGEvaluator` dùng `RetrievalEvaluator` thật khi agent trả về `retrieved_ids`; wire `check_position_bias()` (vốn có sẵn trong code nhưng chưa từng được gọi ở đâu) vào pipeline chính — chạy trên mẫu 10 cặp câu trả lời V1/V2 để đo bias thật thay vì để hàm đó nằm chết. |
| **Consistency Fix** | Phát hiện `reports/summary.json` và `analysis/failure_analysis.md` cũ ghi model/cost không khớp với code thực tế (dấu hiệu bị chỉnh tay thay vì sinh lại từ `main.py`) → chạy lại toàn bộ pipeline thật và viết lại `failure_analysis.md` hoàn toàn dựa trên `benchmark_results.json` mới. |

---

## 2. Kiến thức kỹ thuật đã học được

### MRR (Mean Reciprocal Rank)

MRR = trung bình của `1/rank_i`, trong đó `rank_i` là vị trí 1-indexed của document đúng đầu tiên trong danh sách retrieved.

Ví dụ thật từ benchmark: câu hỏi *"What is VNPAY's customer support hotline number?"* có hit_rate=1.0 (document đúng nằm trong top-6) nhưng MRR chỉ 0.25 (rank thứ 4) — và agent vẫn trả lời sai ("không có thông tin"). Đây là ví dụ sống động cho **trade-off Hit Rate vs MRR**: Hit Rate cao che giấu việc document bị xếp hạng thấp; khi document bị đẩy xuống rank 3-4 trong top-6, chunk cụ thể chứa câu trả lời (hotline number) không lọt vào 2-chunk cap của document đó nữa → agent coi như "không thấy".

### Cohen's Kappa vs Agreement Rate đơn giản — không còn là lý thuyết suông

Agreement Rate đơn giản = `số case đồng ý / tổng case`, với "đồng ý" định nghĩa là delta ≤ 1. Trong benchmark thật, agreement_rate = **100%** (2 judge không bao giờ lệch quá 1 điểm) — nghe rất ấn tượng, nhưng đây chính là điểm yếu mà Kappa vạch trần.

Tôi triển khai Cohen's Kappa thật (`get_cohens_kappa()` trong `llm_judge.py`) dựa trên đồng ý CHÍNH XÁC (exact match trên thang 1-5), trừ đi xác suất đồng ý ngẫu nhiên:

```
po = agreement / n                                    # observed agreement
pe = Σ (row_count[c]/n) × (col_count[c]/n)             # expected by chance
kappa = (po - pe) / (1 - pe)
```

Kết quả thật: **po = 78.2%, pe = 28.9%, kappa = 0.6937**. Đây là mức "substantial agreement" (Landis & Koch), thấp hơn nhiều so với con số "100%" của agreement_rate ngây thơ. Bài học: agreement_rate với ngưỡng lỏng (delta ≤ 1) dễ bị thổi phồng vì cả 2 judge có xu hướng lệch cùng chiều (cùng chấm cao hoặc cùng chấm thấp) — Kappa mới phản ánh đúng "hai judge có thực sự đồng ý ngoài yếu tố ngẫu nhiên hay không".

### Position Bias trong LLM Judge — đo được, không chỉ mô tả

`check_position_bias()` vốn có sẵn trong code từ đầu nhưng **chưa từng được gọi**. Tôi wire nó vào `main.py`, chạy trên mẫu 10 câu hỏi, cho `gpt-4o` so sánh trực tiếp câu trả lời V1 vs V2 rồi đổi thứ tự A/B để kiểm tra tính nhất quán.

Kết quả: **80% (8/10) bị phát hiện position bias** — khi đổi vị trí, judge đổi luôn câu trả lời "thắng" theo vị trí thay vì theo nội dung. Con số cao đáng ngạc nhiên này giải thích tại sao hệ thống chính của lab **không dùng pairwise comparison** mà dùng **absolute scoring độc lập (1-5) cho từng câu trả lời** — cách này không có khái niệm "vị trí" nên tránh được lỗi này hoàn toàn ở cơ chế chính, dù vẫn cần cảnh giác nếu sau này mở rộng sang so sánh trực tiếp nhiều agent (leaderboard, A/B test theo kiểu pairwise).

### Trade-off Chi phí vs Chất lượng Eval

Từ benchmark thật (85 case × 2 version = 170 lượt đánh giá, 2 judges):
- `gpt-4o`: $2.50/1M input, $10/1M output — Judge A, chất lượng cao hơn nhưng đắt gấp ~16 lần `gpt-4o-mini`
- `gpt-4o-mini`: $0.15/1M input, $0.60/1M output — Judge B, rẻ, đủ tốt cho rubric định tính 1-5
- Tổng chi phí thật: **$0.1308** cho 170 lượt đánh giá đầy đủ (agent + 2 judges) = **$0.00077/eval**

**3 cách giảm ~30% chi phí mà không giảm độ chính xác:**
1. **Caching judge results theo (question_hash, answer_hash):** V1 và V2 dùng chung câu hỏi, chỉ khác answer — nhiều câu trả lời giống hệt nhau (đặc biệt các câu "not available") có thể cache để tránh gọi lại judge.
2. **Tiered judging:** chỉ dùng `gpt-4o-mini` (rẻ) cho tất cả case, chỉ escalate lên `gpt-4o` khi điểm nằm ở vùng biên (2-3) hoặc khi cần double-check.
3. **Giảm `max_tokens` của judge** từ 300 xuống ~80 (chỉ cần số điểm + 1 câu lý do ngắn) — giảm trực tiếp chi phí output token, vốn đắt hơn input token 4-40 lần tùy model.

---

## 3. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: Reports cũ không khớp với code — dấu hiệu bị sửa tay

**Vấn đề:** File `reports/summary.json` cũ có trường `note` ghi "Judges: deepseek-v4-pro + gpt-oss-120b" nhưng code hiện tại không hề tạo ra trường `note` này ở đâu cả, và số liệu `estimated_cost_usd` chỉ khớp toán học nếu tính bằng giá `gpt-4o-mini`/`gpt-3.5-turbo` cũ — trong khi `note` lại nói đang dùng model khác. `analysis/failure_analysis.md` cũ cũng mô tả judge khác hẳn (gpt-4o-mini/gpt-3.5-turbo) so với cả 2 nguồn trên.

**Root cause:** Report/analysis từng bị chỉnh tay tại một thời điểm nào đó thay vì luôn được sinh lại bằng `python main.py`, khiến 3 nguồn (code, summary.json, failure_analysis.md) không đồng bộ.

**Giải quyết:** Chạy lại toàn bộ pipeline thật (`python main.py`) sau khi hoàn thiện code, và viết lại `failure_analysis.md` **hoàn toàn dựa trên** `benchmark_results.json` mới — không giữ lại số liệu hay câu chuyện cũ. Bài học lớn nhất: **không bao giờ chỉnh tay output của một pipeline đo lường** — nếu số liệu sai/lỗi thời, phải sửa code và chạy lại, vì báo cáo là artifact đại diện cho hệ thống thật, sửa tay sẽ phá vỡ khả năng tái lập (reproducibility).

### Vấn đề 2: V1 và V2 vốn là cùng một agent — Regression Test không có ý nghĩa

**Vấn đề:** Code gốc gọi `run_benchmark("Agent_V1_Base", judge)` và `run_benchmark("Agent_V2_Optimized", judge)` nhưng cả 2 lần đều khởi tạo `MainAgent()` giống hệt nhau — "V1 vs V2" chỉ là 2 lần chạy cùng 1 agent, chênh lệch điểm số hoàn toàn do nhiễu ngẫu nhiên của model (temperature=0.1), không phản ánh một cải tiến thật nào.

**Giải quyết:** Thêm tham số `version` cho `MainAgent` để V1 (top-k thuần) và V2 (top-k + document-diversity cap) thực sự khác nhau về logic retrieval — biến "so sánh V1 vs V2" từ một phép đo vô nghĩa thành một A/B test thật. Kết quả benchmark thật cho thấy **cải tiến này gần như không tạo khác biệt (delta = −0.01)** — hơi ngược với kỳ vọng ban đầu, nhưng chính vì Regression Gate đo được điều đó nên mới có giá trị thật (xem mục 4 của `failure_analysis.md`).

### Vấn đề 3: Hit Rate đo ở mức document, che giấu lỗi ở mức chunk

**Vấn đề:** Nhiều case có `hit_rate=1.0` (document đúng nằm trong top-k) nhưng agent vẫn trả lời "không có thông tin" — vì `retrieved_ids` chỉ liệt kê document-level, còn agent thực sự chỉ nhìn thấy tối đa 2 chunk của document đó, và chunk chứa chi tiết cần thiết (số liệu, email liên hệ) có thể không nằm trong 2 chunk được chọn.

**Giải quyết (đã áp dụng trong báo cáo):** Không coi Hit Rate=73% là "73% case có đủ thông tin" — phân tích riêng nhóm hit=1.0 để thấy pass rate trong nhóm đó chỉ 77%, không phải 100%. Đây là giới hạn thật của cách đo hiện tại, được ghi rõ trong `failure_analysis.md` thay vì bị bỏ qua.

---

## 4. Nhìn lại và đề xuất cải tiến

**Điều tôi làm tốt:**
- Không chấp nhận số liệu "trông có vẻ ổn" khi phát hiện chúng không khớp với code — ưu tiên chạy lại pipeline thật thay vì tin vào báo cáo có sẵn.
- Biến Regression Test từ phép so sánh giả (2 lần chạy cùng agent) thành A/B test thật có ý nghĩa, kể cả khi kết quả không như kỳ vọng (V2 không tốt hơn V1).
- Thực sự sử dụng `check_position_bias()` và triển khai Cohen's Kappa bằng công thức thật, thay vì chỉ mô tả khái niệm trong reflection mà không có số liệu chứng minh.

**Điều tôi sẽ làm khác nếu có thêm thời gian:**

1. **Semantic/structure-aware Chunking:** tách theo heading, bảng, mục liên hệ (thay vì cắt cố định 800 ký tự) — bằng chứng thật trong `failure_analysis.md` (Case #1) cho thấy đây là root cause lớn nhất, không phải retrieval hay prompting.

2. **Chunk-level Hit Rate:** mở rộng `RetrievalEvaluator` để đo hit rate ở mức chunk chứ không chỉ document — vì phân tích thật cho thấy doc-level Hit Rate lạc quan hơn thực tế khoảng 20 điểm % (77% pass trong nhóm "hit" chứ không phải 100%).

3. **Dense/embedding retrieval đa ngôn ngữ:** đặc biệt quan trọng vì golden set có document tiếng Việt (`be-dieu-khoan-thuong-nhan`) nhưng câu hỏi bằng tiếng Anh — BM25 keyword matching hoàn toàn miss (hit_rate=0) với các câu hỏi dạng này, và đây là nhóm "definition" — nhóm điểm thấp nhất trong toàn bộ benchmark (2.07/5.0).

4. **Lưu lại cả V1 results chi tiết** (không chỉ V2) vào `reports/` — hiện tại chỉ `benchmark_results.json` của V2 được lưu case-level, nên không thể phân tích case-by-case tại sao V1 thắng V2 ở từng câu hỏi cụ thể, chỉ so sánh được ở mức aggregate.

---

*Thực hiện: Lê Hữu Khoa — 2A202600863*
*Ngày: 2026-08-03*
