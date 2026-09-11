# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:11/09/2026**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics: all**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

class_id:      468
class_name:    cab
rank:          1
score:         0.510915
taxonomy_name: ImageNet-1K

- Record này mô tả toàn ảnh như thế nào?

Trong toàn bộ ảnh, model chọn lớp cab là nhãn phù hợp nhất trong danh sách lớp ImageNet-1K mà checkpoint có thể dự đoán

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

Class list được định nghĩa bởi taxonomy/dataset ImageNet-1K. Checkpoint lưu cách ánh xạ giữa chỉ số đầu ra và lớp tương ứng; model chỉ có thể dự đoán các lớp có trong mapping đó
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

Cùng một class_id hoặc cùng một tên lớp có thể mang ý nghĩa khác trong taxonomy khác. Giữ cả ba giúp tái lập kết quả, kiểm tra mapping, tránh nhầm giữa các model/dataset và hỗ trợ chuyển đổi dữ liệu.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

Bải toán là single-classification hay multi-classification. Cách xử lý khi vật thể nhỏ hoặc bị che khuất. Không để 1 label duy nhất
- Vì sao model score không phải ground truth?

score là điểm tin cậy model dành cho lớp cab, không phải nhãn được con người xác nhận. Nên Score chỉ cho biết mức độ model nghiêng về một dự đoán, không chứng minh dự đoán đó đúng

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):


class_name:  person
score:       0.912625
bbox_xyxy:   [385.33, 69.24, 498.92, 348.92]
bbox_width:  113.58 px
bbox_height: 279.68 px
- Diễn giải vị trí box bằng lời:

Box nằm ở vùng giữa lệch phải ảnh:
-Bắt đầu khoảng x = 385, kết thúc x = 499.
-Bắt đầu gần phía trên tại y = 69, kéo xuống y = 349.
-Bao quanh người đang đứng trong khu vực bếp, từ phần đầu đến gần chân.

- So sánh số prediction ở hai threshold:

| Threshold | Số prediction |
|---|---:|
| 0.35 — threshold trong JSON | 11 |
| 0.50 — threshold so sánh | 6 |

Ở mốc 0.50, các predition dưới 0.50 đều bị loại bỏ, kể cả có là prediction bowl với score là 0.499948
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

-Threshold 0.35: bao phủ rộng hơn, giữ lại nhiều vật thể nhỏ hơn, nhưng tăng false positive và số box reviewer phải kiểm tra.
-Threshold 0.50: ít box hơn, reviewer nhanh hơn và kết quả thường sạch hơn, nhưng có thể bỏ sót vật thể nhỏ, bị che khuất hoặc có score thấp.
Đây là trade-off giữa recall/độ bao phủ và chi phí review. Threshold không nên được chọn chỉ dựa trên một ảnh; cần đánh giá trên tập validation có ground truth.
- Đề xuất một quy tắc box chặt:

-Không bao gồm nền hoặc vật thể kế bên.
-Không cắt mất phần object đang nhìn thấy.
-Với object chạm mép ảnh, box được phép chạm mép ảnh.
-Không mở rộng box để bao phủ phần bị che khuất mà không có bằng chứng trực quan.
-Mỗi instance riêng biệt nhận một box riêng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

Guideline cần quy định rõ:
-Có gán nhãn nếu vẫn nhận diện được object dù chỉ thấy một phần hay không.
-Box bao quanh phần nhìn thấy; ghi thêm trạng thái occluded hoặc truncated.
-Nếu object bị cắt bởi mép ảnh, box chạm đúng biên ảnh, không suy đoán phần nằm ngoài ảnh.
-Nếu không chắc đó là một object riêng hay phần của object khác, chuyển sang uncertain/escalate.
-Nên escalation khi occlusion lớn, object quá nhỏ, nhiều box chồng lấn hoặc class có thể bị nhầm lẫn.
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

instance_id:        kitchen-001
class_name:         person
score:              0.899318
polygon_point_count: 348
polygon_xy:         [
  [446, 70],
  [445, 71],
  [444, 71],
  [443, 72],
  [442, 72],
  [441, 73],
  ...
]
- Polygon bổ sung chi tiết gì so với box?

Bounding box chỉ cho biết hình chữ nhật bao quanh object. Polygon mô tả đường biên thực tế hơn:
-ôm theo hình dạng người, thay vì bao gồm toàn bộ hình chữ nhật;
-loại bớt nền nằm trong các góc của box;
-thể hiện được biên cong, phần thắt, phần nhô ra hoặc hình dạng không đều;
- `instance_id` dùng để làm gì và không phải loại ID nào?

instance_id là mã định danh cho một cá thể prediction cụ thể trong sample đó. Nó giúp:
-liên kết class, score, box và polygon cùng một instance;
-phân biệt nhiều object cùng class_name, ví dụ nhiều bowl hoặc nhiều person;
-theo dõi và review/sửa đúng mask.
- Đề xuất một quy tắc biên mask:

Mask nên bao phủ phần pixel nhìn thấy của đúng object và bám sát biên quan sát được:
-không ăn sang nền hoặc object kế bên;
-không bỏ sót phần object nhìn thấy;
-với biên rõ, đặt biên ở ranh giới foreground/background;
-không suy đoán phần bị che khuất;
-không làm mượt quá mức khiến mask mất chi tiết;
-nếu object chạm mép ảnh, mask được phép chạm đúng mép ảnh.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

Guideline cần quy định:
-Vùng mờ nhưng vẫn nhận diện được: mask theo phần có bằng chứng trực quan, không mở rộng theo suy đoán.
-Hai object tiếp xúc: phải tách mask theo biên vật lý; nếu biên không xác định được thì đánh dấu không chắc chắn.
-Object bị che khuất: chỉ mask phần nhìn thấy, đồng thời ghi thuộc tính occluded.
-Object bị cắt mép ảnh: mask dừng tại biên ảnh, đồng thời ghi truncated.
-Nếu không phân biệt được nền với object, hoặc hai instance không thể tách đáng tin cậy, cần escalate để reviewer cấp cao quyết định thay vì tự đoán.
Score 0.899318 chỉ là điểm model; nó không thay thế quyết định về biên mask theo annotation guideline

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một hoặc nhiều nhãn cho toàn ảnh; gồm class_id, class_name, taxonomy_name | Ảnh có nhiều chủ thể; nhãn không mô tả đúng chủ đề chính; lớp gần nghĩa; ảnh không thuộc taxonomy | Chọn nhãn theo guideline; không dùng model score làm ground truth | Kiểm tra nhãn có phù hợp toàn ảnh, đúng taxonomy và nhất quán với các ảnh tương tự hay không |
| Phát hiện vật thể | Mỗi instance gồm class_name/class_id và bounding box xyxy hoặc xywh; tọa độ pixel hoặc chuẩn hóa | Bỏ sót vật thể; box quá rộng/quá chặt; box chồng lấn; vật thể nhỏ, bị che khuất hoặc cắt mép; nhầm lớp | Vẽ box nhỏ nhất bao phủ phần nhìn thấy của object; tạo box riêng cho từng instance | Kiểm tra đủ instance, đúng lớp, box bám sát object, tọa độ hợp lệ và xử lý nhất quán các trường hợp biên |
| Instance segmentation | Mỗi instance gồm instance_id, class_id/class_name và polygon hoặc pixel mask; có thể kèm box | Mask ăn vào nền; bỏ sót biên; biên mờ; hai object tiếp xúc; vùng bị che khuất; polygon tự cắt hoặc quá thô | Vẽ mask theo phần object nhìn thấy, bám sát biên | Kiểm tra mask không lẫn nền/object khác, polygon hợp lệ, instance không bị gộp/tách sai và biên tuân thủ guideline |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:

Chỉ truy cập và xử lý dữ liệu trong hệ thống được phê duyệt; không tự ý sao chép, tải xuống, chia sẻ ra ngoài hoặc dùng dữ liệu cho mục đích khác. Nếu có thông tin nhạy cảm, phải hạn chế truy cập và báo cáo theo quy trình
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

Tech coach; không tiếp tục gán nhãn hay chia sẻ dữ liệu đó

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
