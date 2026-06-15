# Roadmap Nghiên Cứu LeWM-SDN

## Mục Tiêu

Chuyển ý tưởng LeWorldModel từ bài toán điều khiển dựa trên pixel sang bài toán
phát hiện DDoS trong lưu lượng SDN. Mục tiêu nghiên cứu đầu tiên là phát hiện
attack, chưa vội làm mitigation.

Giả thuyết cốt lõi:

> Một mô hình động học ẩn tự giám sát, được huấn luyện trên traffic bình thường,
> sẽ tạo điểm "surprise" cao hơn cho traffic DDoS vì attack phá vỡ động lực học
> thời gian mà mô hình đã học từ network flow.

## Kế Hoạch Theo Phiên Bản

### V0 - Dataset Và Baseline

- Tải CICDDoS2019 bản CSV.
- Kiểm tra phân bố nhãn và các cột feature.
- Xây dựng pipeline tiền xử lý có thể lặp lại.
- Thiết lập baseline giám sát đơn giản:
  - Random Forest hoặc XGBoost nếu môi trường có sẵn.
  - MLP hoặc LSTM nhỏ nếu GPU/RAM cho phép.

Tiêu chí đạt:

- Dataset có thể được load lặp lại ổn định.
- Cách chia train/test được ghi rõ.
- Ghi lại các metric baseline: accuracy, precision, recall, F1, AUROC.

Trạng thái hiện tại:

- Đã tạo `subset_v0.csv` từ hai file đầu: DrDoS_DNS và DrDoS_LDAP.
- Đã chạy baseline nhẹ, kết quả được ghi trong
  [`docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
- Bước tiếp theo là mở rộng subset hoặc bắt đầu viết LeWM-SDN V1 trên dữ liệu
  bảng.

### V1 - LeWM Detection Cho Dữ Liệu Bảng

- Thay encoder pixel bằng encoder chuỗi thời gian cho dữ liệu flow dạng bảng.
- Giữ mô thức chính của LeWM:
  - encoder ánh xạ cửa sổ traffic thành latent embedding;
  - predictor dự đoán latent embedding kế tiếp;
  - SIGReg chống sụp đổ biểu diễn.
- Chỉ dùng label ở bước đánh giá.
- Chấm điểm bất thường bằng lỗi dự đoán latent kế tiếp.

Tiêu chí đạt:

- Model train được trên subset nhỏ bằng RTX 3050 Laptop GPU hoặc CPU.
- Cửa sổ DDoS có anomaly score cao hơn cửa sổ benign.
- Báo cáo AUROC, AUPRC, F1 tại một số ngưỡng và thời gian suy luận.

Trạng thái hiện tại:

- Đã có prototype `numpy` trong `tools/lewm_sdn_v1_numpy.py`.
- Prototype cho thấy prediction error có tín hiệu phân biệt ban đầu, nhưng cần
  threshold calibration và subset giữ thứ tự thời gian tốt hơn.

### V2 - Mô Hình Thời Gian Mạnh Hơn

- Nếu V1 ổn định, thêm encoder chuỗi tốt hơn:
  - Transformer encoder trên cửa sổ flow;
  - temporal convolution;
  - baseline recurrent nhẹ nếu cần.
- So sánh:
  - chỉ dùng prediction loss;
  - prediction loss cộng SIGReg;
  - classifier baseline có giám sát.

Tiêu chí đạt:

- Chứng minh SIGReg có giúp ổn định hoặc tách biệt traffic tốt hơn không.
- Xác định các ca khó như low-rate DDoS hoặc attack có đặc trưng gần benign.

### V3 - Mở Rộng Theo Hướng SDN-Native

- Khi có dataset SDN hoặc trace Mininet phù hợp, chuyển từ flow phẳng sang
  feature có nhận thức topo.
- Thêm graph encoder hoặc encoder tổng hợp theo switch.
- Vẫn ưu tiên detection; chỉ chuyển sang MPC mitigation khi phần detection đã
  đủ đáng tin cậy.

Tiêu chí đạt:

- Chứng minh biểu diễn có nhận thức SDN/topology mang lại lợi ích đo được.
- Chuẩn bị nền tảng cho các action mitigation ở phiên bản sau.

## Cấu Hình Máy Hiện Tại

GPU đã phát hiện: NVIDIA GeForce RTX 3050 Laptop, 4GB VRAM.

Lựa chọn mặc định:

- Bắt đầu với embedding nhỏ: 64 hoặc 128.
- Batch size thận trọng.
- Đọc CSV theo chunk.
- Không load toàn bộ CICDDoS2019 vào RAM cùng lúc.
- Luôn có chế độ subset để chạy thử nghiệm nhanh.

## Chiến Lược Với Repo Hiện Tại

Giữ nguyên đường chạy LeWM gốc, đồng thời thêm tooling và module riêng cho SDN
bên cạnh nó. Các thay đổi đầu tiên chưa nên sửa `train.py`, `jepa.py` hoặc
`module.py` cho đến khi schema dataset được xác nhận.
