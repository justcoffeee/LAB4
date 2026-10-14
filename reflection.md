# Reflection — Representation → neural NLP

## Tra loi cau hoi trong tam: tai sao 2 representation khac nhau?

| | TF-IDF (sparse lexical) | Avg Word2Vec (dense semantic) |
|---|---|---|
| Moi chieu la gi | 1 term cu the (`free`, `not good`) | 1 latent direction, khong dich duoc |
| Thu tu tu | Mat (unigram); bigram giu cuc bo | Mat hoan toan (phep cong giao hoan) |
| Tu hiem/quyet dinh | Gi`u nguyen trong so cao (`num` +8.93) | Chim trong mean ca cau |
| Tu ngoai vocab | Bo qua | Bo qua (OOV ca 2) |
| Phu dinh | Unigram mu; bigram (`not good`) cuu duoc 1 phan | `not` chi la +1 vector → cos(S1,S2)=0.9863, gan nhu khong thay doi |
| Ket qua SMS (test F1) | 0.8962 (LR) / 0.9593 (SVM) | 0.8649 (LR) |

Spam thang ve TF-IDF vi day la bai **lexical**: ke gian lan lo dien qua tu/cum cu the
(`free`, `win`, `text ... to <num>`, `<url>`), khong can hieu nghia sau.
Averaging pha loang dung tin hieu can thiet (Prediction 2 xac nhan: val F1 0.9202
nhỉnh hon TF-IDF 0.9159 nhung test sut 0.8649 < 0.8962 → kem tong quat hon).
Nguoc lai, voi sentiment/topic can hieu dong nghia (`car≈automobile`), TF-IDF thua
(da thay o LAB03) — khong co representation nao thang moi task.

## 3 gioi han cua BoW + static embeddings (dan sang LAB05)

1. **Mat word order / phu dinh**: "good" vs "not good" gan nhu trung vector
   (0.9863). Can mo hinh doc theo thu tu (RNN/LSTM/GRU).
2. **Mot tu = mot vector**: `bank` (ngan hang vs bo song), hay `free` trong
   "free entry" (spam) vs "feel free" (ham) — static vector phai ganh ca 2 nghia.
   Can contextual embedding (attention/Transformer).
3. **Lexical ≠ understanding**: ca TF-IDF lan avg-W2V deu la dem/ cong context,
   khong co cu phap, khong co y do. FN "resent as previous attempt failed due to
   network error" la spam ma BoW bo tay — can hieu ca cau.

## Mach LAB01 → LAB04
LAB01 (sparse search) → LAB02 (probability/LM) → LAB03 (dense semantic) →
LAB04 (representation + classifier + evaluation). Bai hoc lon nhat:
**metric quyet dinh ket luan** (accuracy 95% vo dung khi F1=0), va
**pipeline dung quan trong hon model xin**: split stratified, fit chi tren train
(full-fit vocab 7741 vs train-fit 6327 → +1414 tu leak!), bao cao du P/R/F1 +
confusion matrix thay vi accuracy chay.
