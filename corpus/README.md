# corpus/

Thư mục này chứa văn bản dùng để chấm giọng VietVoice: bộ câu test cố định (đặc tả mục 10).
Kịch bản thu âm không nằm ở đây mà ở `data/scripts/{speaker_id}.tsv` (đặc tả mục 5).

| File | Trạng thái | Nội dung |
|---|---|---|
| `test_sentences_v0.tsv` | **Bản nháp, chưa duyệt** | Bộ câu test v0, khoảng 220 câu, cho giọng Bắc và giọng Nam |

## Quy tắc quan trọng nhất

Câu trong bộ test **không bao giờ** được đưa vào kịch bản thu âm hay dữ liệu huấn luyện
(train/val). Nếu một câu test lọt vào tập train, điểm CER, TER, F0 đo trên nó không còn ý nghĩa.
Script đo độ phủ của P3 phải kiểm tra điều này: không câu nào vừa ở bộ test vừa ở kịch bản thu.
Khi viết kịch bản thu, đừng chép câu từ đây rồi sửa vài chữ.

Mọi câu là văn bản tự viết cho VietVoice, không trích sách, báo, bài hát hay lời người thật.

## Định dạng file

UTF-8 (NFC), mỗi dòng một câu, 4 cột cách nhau bằng TAB, **không có dòng tiêu đề**:

```
id<TAB>dialect<TAB>category<TAB>text
```

- Dòng bắt đầu bằng `#` là chú thích, dòng trống bỏ qua.
- `id`: dạng `v0_<loại>_<số>`, ví dụ `v0_ctt_001`. Không đổi id của câu đã khoá; câu bỏ đi thì
  xoá dòng, không dùng lại id.
- `dialect`: `bac` (chỉ giọng Bắc đọc), `nam` (chỉ giọng Nam đọc), `*` (cả hai giọng đọc).
- `category`: một trong các loại ở bảng dưới.
- `text`: văn bản gốc, như người làm video sẽ gõ. Câu loại `so_va_tieng_anh` có số, đơn vị,
  ký hiệu và từ tiếng Anh nên phải qua chuẩn hoá (P1) trước G2P. Các loại khác chỉ có chữ tiếng
  Việt và dấu `, . ? !`, chạy thẳng `g2p(text.lower(), dialect)` được.

## Các loại câu

| category | Kiểm tra gì | Ghi chú |
|---|---|---|
| `cap_toi_thieu` | Phân biệt 6 thanh | Bộ 6 thanh trong câu khung (ma, má, mà, mả, mã, mạ) và cặp hỏi/ngã, sắc/nặng, huyền/nặng trong câu tự nhiên. Giọng Nam nhập hỏi với ngã: đó là kết quả mong đợi, không phải lỗi |
| `am_tiet_tac` | Âm cuối -p, -t, -c, -ch | Chỉ mang thanh sắc hoặc nặng, ngắn, không bật hơi |
| `mo_dau_nguyen_am` | Tắc thanh hầu đầu âm tiết | Nhiều âm tiết viết bằng nguyên âm: ăn, uống, ở, ấy, ước |
| `tu_vung_mien` | Cách nói tự nhiên của vùng | Câu `bac` dùng từ Bắc (bát, cốc, ô, ngõ, nhé), câu `nam` dùng từ Nam (chén, ly, dù, hẻm, tui, hổng, nghen) |
| `so_va_tieng_anh` | Chuẩn hoá và đọc từ mượn | Giá tiền, phần trăm, ngày tháng, giờ, đơn vị, mã giảm giá, TikTok, video, livestream |
| `cau_hoi_cam_than` | Ngữ điệu cuối câu | Câu hỏi và câu cảm thán ngắn |
| `doan_dai` | Ngữ điệu qua nhiều câu | Đoạn văn đọc khoảng 30–60 giây (170–340 âm tiết): kể chuyện, review, du lịch, tin tức, công thức, quảng cáo |

Câu thường dài 4–25 âm tiết. Nội dung trung tính: không chính trị, tôn giáo, tình dục, không
nói về người thật, không chê thương hiệu thật.

## Các bước duyệt trước khi khoá phiên bản

1. Chủ dự án đọc lại toàn bộ file, sửa câu gượng, câu sai chính tả, câu có nội dung nhạy cảm.
2. Một người Bắc và một người Nam (bản xứ, sống ở vùng đó từ 15 năm trở lên) đọc to từng câu
   của vùng mình và các câu `*`. Đánh dấu câu nào nghe không tự nhiên, từ vùng miền nào không
   đúng cách nói thật.
3. Sửa theo góp ý, chạy lại kiểm tra: mọi câu (trừ `so_va_tieng_anh`) qua G2P cả hai vùng không
   lỗi; câu `so_va_tieng_anh` qua chuẩn hoá P1 không lỗi; độ dài nằm trong khoảng cho phép.
4. Chủ dự án duyệt lần cuối, ghi tên người duyệt và ngày vào chú thích đầu file.
5. Khoá phiên bản: đổi tên thành `test_sentences_v1.tsv`, bỏ chữ "BẢN NHÁP" ở đầu file, commit.
   Từ đó không sửa file đã khoá nữa. Muốn thay đổi thì tạo phiên bản mới (`v2`) và ghi lại trong
   nhật ký thí nghiệm, để kết quả chấm giữa các checkpoint vẫn so được với nhau.

Sau khi khoá, mỗi câu cần một bản ghi thật của người nói để so sánh (đặc tả mục 10). Bản ghi này
thu trong phiên riêng, gắn nhãn `split = test`, không trộn vào giờ thu cho huấn luyện.
