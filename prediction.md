# Prediction (viết TRƯỚC khi chạy model — không sửa sau khi thấy kết quả)

Dataset: SMS Spam Collection (5572 tin nhắn, ham 86.6% / spam 13.4%).

## Prediction 1 — TF-IDF + Logistic Regression có hoạt động tốt không?
- Prediction: Có, F1(spam) ≈ 0.85+, accuracy ≈ 0.97+.
- Reason: spam SMS mang nhiều từ khóa phân biệt mạnh ("free", "win", "prize",
  "text ... to <num>", "<url>") mà TF-IDF bắt trực tiếp qua lexical features;
  LR tuyến tính đủ tách khi features đã tốt.
- Confidence: cao (85%).

## Prediction 2 — Average Word2Vec có chắc chắn tốt hơn TF-IDF không?
- Prediction: Không. Avg-W2V có thể **thua** TF-IDF trên spam (F1 thấp hơn ~5-15 điểm).
- Reason: (1) averaging làm loãng tín hiệu từ khóa hiếm nhưng quyết định
  ("prize" chìm trong mean của cả câu); (2) embedding học trên corpus nhỏ,
  từ spam hiếm → vector noisy; (3) spam phân biệt bằng *từ cụ thể*, không phải
  *ngữ nghĩa*, đúng sở trường lexical của TF-IDF.
- Confidence: trung bình (70%).

## Prediction 3 — Tăng vocabulary size thì accuracy chắc chắn tăng không?
- Prediction: Không chắc; uni+bigram (V 6k→38k) có thể không tăng, thậm chí giảm nhẹ F1.
- Reason: bigram thêm tín hiệu ("not good", "free entry") nhưng feature space thưa hơn,
  nhiều bigram chỉ xuất hiện 1-2 lần → overfit train, chậm train, tốn RAM.
- Confidence: trung bình (65%).

## Prediction 4 — Bỏ stopwords thì performance chắc chắn tăng không?
- Prediction: Không chắc; với spam, giữ stopwords có thể tốt hơn hoặc tương đương.
- Reason: stopwords ("you", "to", "your") mang phong cách spam ("call you", "text to");
  bỏ chúng mất tín hiệu + làm câu ngắn càng ngắn. Lợi ích bỏ stopwords (giảm nhiễu,
  gọn vocab) không universal.
- Confidence: trung bình (60%).

## Prediction 5 — Accuracy và F1 có thể cho kết luận khác nhau không?
- Prediction: Có, chắc chắn trên data imbalance 86/14.
- Reason: accuracy bị chi phối bởi majority class; model thiên về ham vẫn 97%+ accuracy
  trong khi recall(spam) sụt. F1(spam) phản ánh đúng khả năng bắt spam.
  Hai model có thể cùng accuracy nhưng F1 chênh nhau rõ (ví dụ LR vs SVM).
- Confidence: cao (90%).
