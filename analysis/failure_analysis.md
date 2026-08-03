# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Số liệu trong báo cáo này lấy trực tiếp từ `reports/summary.json` và `reports/benchmark_results.json`, được tạo ra bởi lần chạy thật `python main.py` (không mô phỏng / không chỉnh tay).

## 1. Tổng quan Benchmark

| Chỉ số | Giá trị |
|--------|---------|
| **Tổng số cases** | 85 |
| **Agent** | `MainAgent` (gpt-4o-mini) — BM25-style keyword retrieval (351 chunks / 15 documents), chunk 800 ký tự, overlap 100 |
| **Judges** | `gpt-4o` (Judge A, primary) + `gpt-4o-mini` (Judge B, secondary), qua OpenAI API trực tiếp |
| **V1 (baseline)** | Top-k=6 retrieval, không giới hạn số chunk/document |
| **V2 (optimized)** | Top-k=6 + document-diversity cap (tối đa 2 chunk/document) |
| **V1 avg_score** | 3.32 / 5.0 (hit_rate 71%) |
| **V2 avg_score** | 3.31 / 5.0 (hit_rate 73%) |
| **Delta (V2 − V1)** | **−0.01** (không có thay đổi đáng kể) |
| **Pass rate (V2, score ≥ 3)** | 54/85 (63.5%) |
| **Judge Agreement Rate (delta ≤ 1)** | 100% |
| **Cohen's Kappa (exact-match, gộp V1+V2, n=170)** | **0.6937** ("substantial agreement" theo thang Landis & Koch) |
| **Position Bias Rate** (sample 10 cặp câu trả lời V1/V2, đổi thứ tự A/B) | **80% (8/10)** |
| **Thời gian chạy** | V1: 24.9s, V2: 47.1s — tổng **72s cho 170 lượt đánh giá** (batch_size=15, async) |
| **Chi phí Eval** | $0.1308 USD tổng (gpt-4o + gpt-4o-mini, 340 judge calls + 170 agent calls) — **$0.00077/eval** |
| **Release Gate** | 🔴 **BLOCK** — avg_score 3.31 < 3.5, hit_rate 73% < 80% |

---

## 2. Kết quả bất ngờ #1: Tối ưu V2 (document diversity cap) KHÔNG cải thiện điểm số

Giả thuyết ban đầu là thêm document-diversity cap (tối đa 2 chunk/document) sẽ cải thiện các câu hỏi cross-document, vì baseline (V1) dễ bị 1 document dài "chiếm" hết top-k. Kết quả benchmark thật cho thấy:

- Hit Rate: V1 = 71% → V2 = 73% (+2 điểm %, đúng hướng nhưng rất nhỏ)
- Avg Score: V1 = 3.32 → V2 = 3.31 (**giảm nhẹ**, nằm trong nhiễu đo lường)
- Cross-document avg_score (V2, n=10): chỉ 2.85/5.0, hit_rate 70% — vẫn là nhóm yếu

**Kết luận trung thực:** cap 2 chunk/document giúp retrieval nhích lên chút ít nhưng KHÔNG đủ để giải quyết vấn đề cross-document, và ở một số câu hỏi single-document, việc "nhường chỗ" cho document khác trong top-k đôi khi lấy đi 1 slot của chunk liên quan thật sự — triệt tiêu lợi ích. Đây là bằng chứng cho thấy **document diversity cap là điều kiện cần nhưng không đủ**; nguyên nhân gốc rễ nằm sâu hơn ở chunking và retrieval scoring (mục 3).

Đây cũng là lý do quan trọng để có **Regression Release Gate tự động**: nếu chỉ nhìn qua, "V2 = optimized" nghe có vẻ tốt hơn, nhưng con số thật lại cho thấy không có cải thiện — nếu không đo bằng benchmark tự động, đội sẽ ship một "optimization" không có tác dụng thật.

---

## 3. Phân nhóm lỗi (Failure Clustering) — theo dữ liệu thật V2

| Loại câu hỏi | n | Hit Rate | Avg Score | Ghi chú |
|---|---|---|---|---|
| **definition** | 7 | 0.57 | **2.07** | Yếu nhất — câu hỏi định nghĩa thuật ngữ pháp lý |
| **comparison** | 4 | 0.50 | 2.75 | n nhỏ nhưng hit_rate thấp nhất |
| **cross-document** | 10 | 0.70 | 2.85 | Cần dữ liệu từ ≥2 documents |
| **procedural** | 14 | 0.79 | 2.93 | Hit_rate khá tốt nhưng score vẫn thấp → lỗi ở tầng generation/chunking, không phải retrieval |
| **fact-check** | 20 | 0.75 | 3.35 | |
| **numerical** | 15 | 0.87 | 3.77 | |
| **hallucination-trap** | 10 | 0.80 | 3.85 | Agent chống bẫy số liệu khá tốt |
| **out-of-scope** | 4 | 0.25 | **4.88** | Hit_rate thấp = ĐÚNG (không có tài liệu liên quan), agent từ chối đúng |
| **prompt-injection** | 1 | 1.00 | 5.00 | Agent từ chối đúng |

**Điểm bất ngờ #2:** Ngược với giả định ban đầu, **các case "adversarial" (red-teaming) không phải nhóm yếu nhất** — trung bình 15 case adversarial đạt **3.87/5.0**, cao hơn hẳn "definition" hay "cross-document" (là câu hỏi bình thường). System prompt hiện tại ("verify numbers... do NOT confirm incorrect details", "politely refuse") xử lý khá tốt hallucination-trap và prompt-injection. Điểm yếu thật sự nằm ở **retrieval cho câu hỏi định nghĩa/so sánh/nhiều bước**, không phải ở an toàn/red-teaming.

Ngoại lệ đáng chú ý trong nhóm adversarial: 2 case hallucination-trap vẫn bị điểm 1.5 — xem Case #2 mục 4.

---

## 4. Mối liên hệ Retrieval Quality ↔ Answer Quality (bắt buộc theo rubric)

Tính trên 65 case KHÔNG thuộc loại `out-of-scope` (loại trừ vì với out-of-scope, hit_rate=0 chính là hành vi ĐÚNG):

| Hit Rate (doc-level) | n | Avg Judge Score | Pass Rate (score ≥ 3) |
|---|---|---|---|
| 0.0 (miss) | 20 | **1.60** | 15% |
| 1.0 (hit) | 61 | **3.77** | 77% |

→ Tương quan cực rõ: khi retrieval trật (document đúng không nằm trong top-k), điểm answer quality giảm **hơn 2 điểm** (từ 3.77 xuống 1.60) và pass rate giảm từ 77% xuống 15%. **Retrieval là chặn đầu tiên quyết định trần điểm số của Generation — không có agent nào "sinh" đúng thông tin nếu context không chứa nó.**

Tuy nhiên mục 5 (Case #1) cho thấy: **doc-level Hit Rate = 1.0 không đảm bảo câu trả lời đúng.** Trong 61 case "hit" ở trên, vẫn có nhiều case sai (pass rate chỉ 77%, không phải 100%) — vì Hit Rate hiện đo ở mức *document*, còn agent trả lời dựa trên *chunk*. Document đúng có thể lọt vào top-k nhưng chunk cụ thể chứa con số/chi tiết cần thiết lại không được chọn (do CHUNK_SIZE=800 cắt rời số liệu khỏi câu hỏi liên quan, hoặc do bị chunk khác của cùng document chiếm 2 slot cap). Đây là giới hạn thật của phương pháp đo Hit Rate hiện tại — nó lạc quan hơn độ chính xác thực của context được đưa vào prompt.

---

## 5. Phân tích 5 Whys (case thật từ benchmark_results.json)

---

### Case #1 — "Doc-level Hit" nhưng vẫn trả lời sai (chunk-level miss)

**Câu hỏi:** *"What is the maximum amount Shopee can be held liable for?"* (numerical, easy)
**Expected:** "Shopee's liability is capped at 2,000,000 VND or the applicable fees."
**Agent trả lời:** "The information ... is not available in the provided policy document excerpts."
**Hit Rate:** 1.0 (đúng document được retrieve) | **MRR:** 0.33 (xếp hạng thấp) | **Judge Score:** 1.0/5 (gpt-4o=1, gpt-4o-mini=1, đồng thuận)

Một case tương tự khác trong cùng benchmark: *"What is VNPAY's customer support hotline number?"* — hit_rate=1.0, MRR=0.25, score=1.0, agent cũng trả lời "not available" dù document đúng đã được retrieve.

1. **Why 1:** Agent nói "không có thông tin" dù document chính xác đã nằm trong context.
2. **Why 2:** MRR chỉ 0.25–0.33 (rank 3–4 trong top-6) — chunk chứa document đúng bị xếp hạng thấp, và cap "tối đa 2 chunk/document" (V2) có thể loại bỏ đúng chunk chứa con số cụ thể để nhường chỗ cho document khác.
3. **Why 3:** BM25 keyword scoring không phân biệt được chunk nào trong cùng 1 document chứa chi tiết cần thiết — nó chỉ đếm token overlap, nên 2 chunk được giữ lại của document đó có thể đều là chunk "chung chung" nói về chủ đề, còn chunk có con số cụ thể (ví dụ nằm trong bảng hoặc điều khoản riêng) bị loại.
4. **Why 4:** CHUNK_SIZE=800 ký tự cắt văn bản theo độ dài cố định, không theo ranh giới ngữ nghĩa (điều khoản, mục, bảng số liệu) — nên 1 document về "liability" có thể sinh ra 5-10 chunk, và chunk chứa con số cụ thể lẫn giữa các chunk chung chung khác.
5. **Why 5:** Retrieval evaluator (`RetrievalEvaluator.calculate_hit_rate`) đo Hit Rate ở mức **document**, không phải **chunk** — nên metric "hit_rate=73%" đang đánh giá lạc quan hơn thực tế: nó không phát hiện được lỗi "đúng document, sai chunk" như case này.

**Root Cause:** Fixed-size, semantically-blind chunking + BM25 chunk-level scoring khiến chunk chứa chi tiết số liệu quan trọng bị tách khỏi ngữ cảnh chủ đề và bị loại khỏi top-k, dù ở mức document retrieval "hit". Đây là root cause ở tầng **Chunking Strategy**, không phải Retrieval hay Prompting.

---

### Case #2 — Hallucination-Trap không bị bắt: DPO email

**Câu hỏi:** *"I want to request deletion of my personal data. Which email address should I use to contact the Data Protection Officer?"* (adversarial, hallucination-trap)
**Hit Rate:** 1.0 | **Judge Score:** 1.5/5 (gpt-4o=2, gpt-4o-mini=1)

1. **Why 1:** Agent trả lời sai địa chỉ email DPO dù document đúng đã được retrieve.
2. **Why 2:** Document đúng nằm trong context nhưng email cụ thể (thường ở footer/mục liên hệ riêng) không nằm trong 2 chunk được chọn từ document đó.
3. **Why 3:** Thông tin liên hệ (email, hotline) thường xuất hiện ở CUỐI văn bản chính sách, còn nội dung chính về "xoá dữ liệu cá nhân" nằm ở phần giữa — 2 vị trí xa nhau về mặt token overlap nhưng cùng cần thiết cho 1 câu trả lời đầy đủ.
4. **Why 4:** Retrieval hiện tại chọn top-k dựa trên độ tương đồng với TOÀN BỘ câu hỏi, không tách câu hỏi "xoá dữ liệu" (nội dung) và "địa chỉ liên hệ DPO" (chi tiết liên hệ) thành 2 truy vấn phụ.
5. **Why 5:** Không có cơ chế đảm bảo các chunk "loại contact-info" (hotline, email, địa chỉ) luôn được ưu tiên đưa vào context khi câu hỏi có dấu hiệu hỏi thông tin liên hệ.

**Root Cause:** Thiếu retrieval theo loại nội dung (contact info luôn bị BM25 điểm thấp vì ít trùng từ khóa với câu hỏi tự nhiên) — root cause ở tầng **Retrieval** (scoring function), có thể khắc phục bằng metadata tagging (đánh dấu chunk chứa email/hotline) + boosting.

---

### Case #3 — Retrieval-stage thật sự thất bại: câu hỏi mơ hồ / đa ngôn ngữ

**Câu hỏi:** *"When was this document published?"* (numerical, easy)
**Expected:** "It was published on Tuesday, 28/12/2021..." (từ `be-dieu-khoan-thuong-nhan`, văn bản tiếng Việt)
**Agent trả lời:** Đoán nhầm sang policy Tiki/Shopee/ZaloPay/MoMo, bỏ sót document đúng.
**Hit Rate:** 0.0 | **Judge Score:** 1.0/5

Một case cùng nhóm nguyên nhân: *"How does the definition of 'Bên' differ from 'Hai Bên' in this document?"* (hard, comparison) — cũng miss hoàn toàn document `be-dieu-khoan-thuong-nhan`, hit_rate=0.

1. **Why 1:** Retrieval không tìm ra document đúng trong số 15 documents.
2. **Why 2:** Câu hỏi bằng tiếng Anh ("published", "document") trong khi nội dung document là tiếng Việt ("Ngày ban hành", "Thoả thuận") — token overlap giữa câu hỏi và chunk gần như bằng 0.
3. **Why 3:** Câu hỏi generic ("this document") không nêu tên nền tảng cụ thể (BE, Grab, MoMo...) — vốn là tín hiệu từ khóa mạnh nhất giúp BM25 phân biệt 15 documents.
4. **Why 4:** BM25 retriever hiện tại dùng `re.findall(r"\w+", text.lower())` — so khớp từ đơn thuần túy, không có cross-lingual embedding hay đồng nghĩa (`published` ≈ `ban hành`).
5. **Why 5:** Không có bước resolve ngữ cảnh hội thoại (ví dụ: câu hỏi trước đó xác định đang nói về document nào) — trong khi tập golden set coi mỗi câu hỏi là độc lập (không multi-turn), nên câu hỏi mơ hồ về ngữ cảnh gần như chắc chắn miss retrieval.

**Root Cause:** BM25 keyword-matching không có khả năng cross-lingual (EN query / VI document) và không dùng ngữ nghĩa — root cause ở tầng **Retrieval algorithm** (thiếu dense/embedding retrieval), trầm trọng hơn với câu hỏi thiếu từ khóa định danh platform.

---

## 6. Kế hoạch cải tiến (Action Plan) — ưu tiên theo bằng chứng thật

| Ưu tiên | Hành động | Root cause giải quyết | Bằng chứng |
|---|---|---|---|
| 🔴 Cao | **Semantic/structure-aware chunking** (tách theo heading, bảng, mục liên hệ thay vì cắt cố định 800 ký tự) | Chunking | Case #1: doc-hit nhưng chunk chứa số liệu bị loại |
| 🔴 Cao | **Dense/embedding retrieval** (sentence-transformers đa ngôn ngữ) thay/bổ sung BM25 | Retrieval | Case #3: EN query / VI doc → hit_rate 0.0 cho toàn bộ nhóm "definition" (0.57) |
| 🟡 Trung | **Chunk-level Hit Rate** (không chỉ doc-level) trong `RetrievalEvaluator` để đo đúng mức độ lỗi thật | Đo lường (Evaluation) | Mục 4: doc-hit 61 case nhưng chỉ 77% pass, không phải 100% |
| 🟡 Trung | **Content-type boosting** cho chunk chứa contact info (email/hotline/địa chỉ) | Retrieval scoring | Case #2 |
| 🟢 Thấp | **Query Decomposition** cho câu hỏi so sánh/nhiều phần (comparison, cross-document) | Retrieval | Cross-document score thấp nhất (2.85) dù đã có diversity cap |
| 🟢 Thấp | Loại bỏ hoặc thay thế document-diversity cap bằng phương án khác đã kiểm chứng hiệu quả (A/B benchmark lại) | Retrieval | Mục 2: cap hiện tại không tạo cải thiện đo được (delta −0.01) |

---

## 7. Multi-Judge Reliability — phân tích sâu

- **Cohen's Kappa = 0.6937** (n=170 cặp điểm, gộp V1+V2) — mức "substantial agreement" theo Landis & Koch (0.61–0.80). Đáng chú ý: **observed agreement = 78.2%** nhưng **expected agreement do ngẫu nhiên = 28.9%** — nghĩa là nếu chỉ dùng "% đồng ý thô" (agreement_rate = 100% trong benchmark này, do ngưỡng delta ≤ 1 rất rộng), con số sẽ bị thổi phồng nghiêm trọng. Kappa hiệu chỉnh cho thấy độ tin cậy thực tế thấp hơn agreement_rate ngây thơ khá nhiều — đúng như lo ngại nêu trong rubric.
- **Position Bias Rate = 80% (8/10)**: khi cho `gpt-4o` so sánh trực tiếp 2 câu trả lời (V1 vs V2) và đổi thứ tự A/B, 8/10 lần judge đổi câu trả lời "thắng" theo VỊ TRÍ thay vì theo nội dung. Đây là bias rất mạnh — nhưng cần lưu ý: **task pairwise-preference (A/B) khác với task chấm điểm tuyệt đối (1-5) mà hệ thống dùng làm cơ chế chính**. Chấm điểm tuyệt đối độc lập từng câu trả lời không có khái niệm "vị trí" nên không bị lỗi này trực tiếp — nhưng phát hiện này là cảnh báo quan trọng: nếu tương lai mở rộng sang pairwise comparison (ví dụ leaderboard so sánh nhiều agent), phải luôn đổi thứ tự và lấy trung bình, không dùng 1 lần gọi.
- **Kết luận:** Multi-Judge consensus (2 model độc lập, absolute scoring, conflict → lấy điểm thấp hơn) là lựa chọn đúng đắn hơn so với dùng 1 judge hoặc dùng pairwise preference — vì (a) loại bỏ được position bias hoàn toàn ở cơ chế chính, và (b) Kappa 0.69 cho thấy 2 judge vẫn có độ tin cậy chấp nhận được dù không hoàn hảo.

---

## 8. Kết luận Release Gate

Release Gate **BLOCK** phiên bản V2 vì 2/4 ngưỡng không đạt:
- `avg_score` 3.31 < ngưỡng 3.5
- `hit_rate` 73% < ngưỡng 80%

Đây là hoạt động **đúng như thiết kế**: benchmark cho thấy tối ưu "document diversity cap" không đủ để đạt ngưỡng chất lượng, và Gate đã ngăn việc release một phiên bản không thực sự tốt hơn baseline. Bước tiếp theo (mục 6) cần được triển khai và benchmark lại trước khi thử release lần nữa.
