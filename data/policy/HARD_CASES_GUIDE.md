# Hướng dẫn thiết kế Hard Cases cho AI Evaluation

Để bài lab đủ độ khó cho nhóm 6 người, các bạn cần thiết kế các test cases có tính thử thách cao:

### 1. Adversarial Prompts (Tấn công bằng Prompt)
- **Prompt Injection:** Thử lừa Agent bỏ qua context để trả lời theo ý người dùng.
- **Goal Hijacking:** Yêu cầu Agent thực hiện một hành động không liên quan đến nhiệm vụ chính (ví dụ: đang là hỗ trợ kỹ thuật nhưng yêu cầu viết thơ về chính trị).

### 2. Edge Cases (Trường hợp biên)
- **Out of Context:** Đặt câu hỏi mà tài liệu không hề đề cập. Agent phải biết nói "Tôi không biết" thay vì bịa chuyện (Hallucination).
- **Ambiguous Questions:** Câu hỏi mập mờ, thiếu thông tin để xem Agent có biết hỏi lại (clarify) không.
- **Conflicting Information:** Đưa ra 2 đoạn tài liệu mâu thuẫn nhau để xem Agent xử lý thế nào.

### 3. Multi-turn Complexity
- **Context Carry-over:** Câu hỏi thứ 2 phụ thuộc vào câu trả lời thứ 1.
- **Correction:** Người dùng đính chính lại thông tin ở giữa cuộc hội thoại.

### 4. Technical Constraints
- **Latency Stress:** Yêu cầu Agent xử lý một đoạn văn bản cực dài để đo giới hạn latency.
- **Cost Efficiency:** Đánh giá xem Agent có đang dùng quá nhiều token không cần thiết cho các câu hỏi đơn giản không.
