# LAB 04 — Calculations (tính tay, không dùng Python)

## Bài 1 — Binary classification (logit → sigmoid)

Cho x=[1,2,0], w=[0.5,−0.2,0.4], b=0.1.

```
z = wᵀx + b = 0.5*1 + (−0.2)*2 + 0.4*0 + 0.1
  = 0.5 − 0.4 + 0 + 0.1 = 0.2
σ(z) = 1/(1+e^(−0.2)) = 1/(1+0.8187) ≈ 0.5498
```

Vậy P(y=1|x) ≈ **0.55**. Đây chính là công thức Logistic Regression dùng ở
core implementation: logit z là tổ hợp tuyến tính của features, sigmoid
squash về (0,1) thành xác suất.

## Bài 2 — Prediction threshold

P(y=1|x)=0.72.

- τ=0.5: 0.72 ≥ 0.5 → dự đoán **class 1 (positive/spam)**.
- τ=0.8: 0.72 < 0.8 → dự đoán **class 0 (negative/ham)**.

Ảnh hưởng lên precision/recall: tăng τ → model "khó tính" hơn khi gán positive
→ ít false positive → **precision tăng**, nhưng bỏ sót positive thật
(nhiều false negative) → **recall giảm**. Ngược lại, giảm τ → recall tăng,
precision giảm. Chọn τ tùy chi phí lỗi: spam detection thường chịu FN hơn FP
(chặn nhầm tin nhắn quan trọng nguy hiểm hơn lọt 1 spam), nên τ cao hợp lý;
chẩn đoán bệnh thì ngược lại (cần recall cao).

## Bài 3 — Confusion matrix

TP=80, FN=20, FP=10, TN=90 (N=200).

```
Precision = TP/(TP+FP) = 80/90 ≈ 0.8889
Recall    = TP/(TP+FN) = 80/100 = 0.8000
F1        = 2PR/(P+R) = 2*0.8889*0.8/1.6889 ≈ 0.8421
Accuracy  = (80+90)/200 = 0.85
```

Ý nghĩa với bài toán spam (positive=spam):
- False positive (10): tin nhắn thường bị gắn spam → người dùng **mất tin quan trọng**
  (ví dụ mã OTP, tin nhắn ngân hàng). Rất tai hại, làm mất niềm tin.
- False negative (20): spam lọt vào inbox → phiền toái, tốn thời gian, rủi ro lừa đảo
  (link độc, "free entry... Text FA to 87121").
Precision thấp = quấy rầy người dùng; recall thấp = lưới lọc thủng.

## Bài 4 — Accuracy trap

Dataset 950 negative / 50 positive. Model "mọi sample → negative":
accuracy = 950/1000 = **95%**, nhưng:

```
Recall(positive) = 0/50 = 0
Precision(positive) = undefined (0/0) → 0
F1(positive) = 0
```

Model **vô dụng** cho mục tiêu bài toán (không bắt được spam nào) dù accuracy 95%.
Phải xem thêm: precision/recall/F1 **từng class** (đặc biệt minority class),
confusion matrix, balanced accuracy, PR-curve. Đây chính là tình huống của
SMS Spam (ham 86.6% / spam 13.4%): baseline "luôn đoán ham" đã 86.6% accuracy
mà F1(spam)=0 — vì vậy lab này cấm chỉ báo cáo accuracy.
