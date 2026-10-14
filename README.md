# LAB 04 — Text Classification (SMS Spam)
Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng | Học kỳ I - 2026

## 1. Cấu trúc nộp
```
lab04/
├── README.md
├── calculations.md        # Bài 1-4 (logit/sigmoid, threshold, confusion, accuracy trap)
├── prediction.md          # 5 predictions trước experiment
├── preprocessing.py       # clean_text, load_sms, stratified_split, fit_tfidf (chỉ fit trên train)
├── classification.ipynb   # toàn bộ thí nghiệm
├── sms.tsv                # SMS Spam Collection 5572 tin (nguồn: justmarkham/pycon-2016-tutorial)
├── w2v_vectors.npy / w2v_vocab.json  # Skip-gram NEG dim=50 train riêng trên train-corpus
├── results.csv            # metrics thật (val + test)
├── confusion_matrix.png   # TF-IDF+LR trên test: [[719,5],[17,95]]
├── error_analysis.md      # 3 đúng + 3 sai (ví dụ thật)
└── reflection.md
```

## 2. Cách chạy
```bash
python lab04/preprocessing.py          # kiểm tra data + split
# mở lab04/classification.ipynb chạy từng cell
```
Word2Vec (Skip-gram NEG numpy-only, kế thừa LAB03 vì Python 3.14 chưa có wheel gensim):
dim=50, window=5, min_count=5, epochs=5, negative=5, 60k sampled pairs (seed=42,
~150s CPU). Vector đã lưu sẵn để notebook chạy nhanh; train lại xem `exp_w2v.py`
(đã xóa sau khi lấy số — công thức giữ nguyên trong README này).

## 3. Kết quả chính (test, spam=positive)
| Model | Acc | P | R | F1 |
|---|---|---|---|---|
| TF-IDF uni + LR (baseline) | 0.9737 | 0.9500 | 0.8482 | 0.8962 |
| Avg-Word2Vec + LR | 0.9641 | 0.8727 | 0.8571 | 0.8649 |
| TF-IDF uni + SVM | 0.9892 | 0.9725 | 0.9464 | **0.9593** |
| TF-IDF uni+bi + LR | 0.9725 | 0.9406 | 0.8482 | 0.8920 (V 6k→38k, không tăng) |
| TF-IDF bỏ stopwords + LR | 0.9701 | 0.9307 | 0.8393 | 0.8826 (giữ stopwords tốt hơn) |

Điểm đáng chú ý: cos(avg("this movie is good"), avg("... not good"))=0.9863
→ averaging mù trước phủ định. Leakage demo: fit TF-IDF trên full data cho
vocab 7741 vs 6327 khi chỉ fit train (+1414 từ test lọt vào IDF).

## 4. AI assistance statement
- Tool: Muse Spark (opencode agent) + sklearn/matplotlib.
- Purpose: debugging, giải thích lỗi, gợi ý dùng sklearn, visualization.
- Generated: khung pipeline/preprocessing/notebook; Modified: corpus chọn SMS Spam,
  siêu tham số, diễn giải metrics và nguyên nhân lỗi viết thủ công từ output thật.
- Verified: mọi số trong results.csv copy từ log chạy (seed=42); confusion matrix
  đối chiếu TN+FP=724 ham, FN+TP=112 spam khớp test split.
