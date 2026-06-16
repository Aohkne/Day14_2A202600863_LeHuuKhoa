# Tiêu chí Chấm điểm EXPERT LEVEL - Lab Day 14

Bài lab dành cho nhóm (4-6 người) được đánh giá trên thang điểm 100 theo tiêu chuẩn AI Engineering chuyên nghiệp.

---

## 👥 1. Điểm Nhóm (Tối đa 60 điểm)

| Hạng mục | Tiêu chí | Điểm |
| :--- | :--- | :---: |
| **Retrieval Evaluation** | - Tính toán thành công Hit Rate & MRR cho ít nhất 50 test cases.<br>- Giải thích được mối liên hệ giữa Retrieval Quality và Answer Quality. | 10 |
| **Dataset & SDG** | - Golden Dataset chất lượng (50+ cases) với mapping Ground Truth IDs.<br>- Có các bộ "Red Teaming" phá vỡ hệ thống thành công. | 10 |
| **Multi-Judge consensus** | - Triển khai ít nhất 2 model Judge (ví dụ GPT + Claude).<br>- Tính toán được độ đồng thuận và có logic xử lý xung đột tự động. | 15 |
| **Regression Testing** | - Chạy thành công so sánh V1 vs V2.<br>- Có logic "Release Gate" tự động dựa trên các ngưỡng chất lượng. | 10 |
| **Performance (Async)** | - Toàn bộ pipeline chạy song song cực nhanh (< 2 phút cho 50 cases).<br>- Có báo cáo chi tiết về Cost & Token usage. | 10 |
| **Failure Analysis** | - Phân tích "5 Whys" cực sâu, chỉ ra được lỗi hệ thống (Chunking, Ingestion, v.v.). | 5 |

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục | Tiêu chí | Điểm |
| :--- | :--- | :---: |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).<br>- Chứng minh qua Git commits và giải trình kỹ thuật. | 15 |
| **Technical Depth** | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.<br>- Hiểu về trade-off giữa Chi phí và Chất lượng. | 15 |
| **Problem Solving** | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp. | 10 |

---

## 📋 Quy trình nộp bài
1. Chạy `python check_lab.py` để đảm bảo mọi module hoạt động.
2. Nộp Repository link kèm file `reports/summary.json` có chứa cả kết quả Regression.

> [!CAUTION]
> **Điểm liệt:** Nếu nhóm chỉ sử dụng 1 Judge đơn lẻ hoặc không có Metrics đánh giá Retrieval, điểm tối đa phần Nhóm sẽ bị giới hạn ở mức 30 điểm.
