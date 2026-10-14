# Error analysis — model TF-IDF unigram + LR, test set (836 tin: ham 724 / spam 112)
# Baseline: acc=0.9737, P(spam)=0.9500, R(spam)=0.8482, F1=0.8962 (xem results.csv).
# Confusion: TN=719, FP=5, FN=17, TP=95.

## 3 correctly classified

### 1. TP — spam kinh dien, trung tu khoa manh
- Text: "You have won a guaranteed 200 award or even 1000 cash to claim UR award call free on 08000407165..."
- True=spam, Pred=spam.
- Analysis: hoi tu dong loat features weight cao nhat: `num` (8.93), `free`, `call`,
  `to`, `your`, `stop`. Logit z >> 0 → dung chac chan. Day la vung model manh nhat:
  spam template truyen thong.

### 2. TP — dating/XCHAT spam
- Text: "Dear 0776xxxxxxx U've been invited to XCHAT... Txt CHAT to 86688..."
- True=spam, Pred=spam.
- Analysis: `txt/text`, `to`, `num` kich hoat. Du`u've`, `xchat` hiem, cum so + lenh
  "Txt ... to ..." du de keo ve spam. Minh hoa tai sao lexical features thang
  semantic o bai nay.

### 3. TN — ham hoi thoai ngan
- Text: "Yes.mum lookin strong:)"
- True=ham, Pred=ham.
- Analysis: toan tu ham (`my/i/ok/u/come` weight am), khong co `num/url/lenh`.
  z am sau → ham. Model xu ly tot ham ngan khi khong co trigger.

## 3 incorrectly classified

### 1. FN — spam "sach", khong tu khoa (nguy hiem nhat)
- Text: "This msg is for your mobile content order It has been resent as previous attempt failed due to network error Queries to customersqueries@netvision.uk.com"
- True=spam, Pred=ham.
- Category: **lexical ambiguity / thieu tin hieu + domain-specific**.
- Analysis: khong co `free/win/prize/call`, giong van phong "resent/network error/queries".
  `mobile/your` thi co nhung bi dan ham (`to/for/has/been`) ap dao. Loai loi dac trung
  cua BoW: khong hieu *y do* ("resent due to network error" la co lua kinh dien).

### 2. FN — quang cao "lich su" co nut STOP
- Text: "Orange brings you ringtones from all time Chart Heroes, with a free hit each week! ... reply STOP."
- True=spam, Pred=ham.
- Category: **data noise / phan phoi lech**: tu `stop/reply/free` vo`n la +spam,
  nhung "Orange/brings/ringtones/chart" hiem trong train-spam, trong khi van phong
  than thien ("brings you ... each week") giong ham. Model hoc `stop`=+2.33 nhung
  tong trong so van am. Minh hoa gioi han tuyến tinh: khong mo hinh hoa tuong tac
  "free hit each week + reply STOP = quang cao".

### 3. FP — ham cuc ngan bi `num` danh lua (x2)
- Text: "U 2." / "S..antha num corrct dane"
- True=ham, Pred=spam.
- Category: **short text + rare word/noise**.
- Analysis: sau clean chi con 1-3 tokens (`u <num>`), trong do `num` weight +8.93
  (cao nhat model!). Cau ngan → TF-IDF khong co context de can bang → 1 feature
  quyet dinh tat ca. Day la mat trai cua keyword manh: chi 1 so dien thoai trong
  tin nhan vo toi cung du bi chan (FP tai hai: mat tin quan trong).

## Tong ket nguyen nhan (map voi de)
- **Short text**: 2/3 loi FP la tin <10 ky tu; BoW can do dai de on dinh.
- **Lexical ambiguity**: spam "sach" dung tu thuong → BoW mu.
- **Rare/noise word**: "S..antha", "Bloomberg" hiem → OOV/weights kem.
- **Class imbalance (86.6/13.4)**: model thien ham → FN=17 >> FP=5 (recall 0.848 < precision 0.95).
- **Negation/word-order**: `not` khong co tac dung trong ca TF-IDF lan avg-W2V
  (cos S1/S2 = 0.9863) — BoW cong vector, phu dinh bi chìm.
- Huong fix: bigram giup it (F1 giam 0.896→0.892 do thua), thu thap them spam "sach",
  hieu chinh threshold theo chi phi FP/FN, chuyen sang sequence/contextual model (LAB05+).
