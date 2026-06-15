# Nhật Ký Thí Nghiệm

## V0 Baseline Trên `subset_v0.csv`

Ngày chạy: 2026-06-14

Dataset:

- Nguồn: CICDDoS2019 CSV.
- File subset: `C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv`
- Số dòng: 15,000.
- Feature numeric: 82.
- Phân bố nhãn nhị phân:
  - ATTACK: 10,000.
  - BENIGN: 5,000.
- Nhãn attack trong subset:
  - DrDoS_DNS: 5,000.
  - DrDoS_LDAP: 5,000.

Lệnh:

```powershell
python tools\baseline_cicddos2019.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Kết quả:

| Baseline | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| benign-zscore-anomaly | 0.8947 | 0.9667 | 0.8720 | 0.9169 | 0.9862 | 0.9909 |
| nearest-centroid-binary | 0.9893 | 0.9986 | 0.9853 | 0.9919 | 0.9950 | 0.9980 |

Ghi chú:

- Đây là mốc kiểm tra pipeline, chưa phải kết luận nghiên cứu.
- Subset hiện chỉ gồm benign, DrDoS_DNS và DrDoS_LDAP nên bài toán còn tương đối
  dễ.
- Kết quả cao là tín hiệu pipeline ổn, nhưng cần mở rộng sang nhiều attack type
  hơn và split khó hơn trước khi so sánh với LeWM-SDN.

## V1 Prototype: Latent Prediction Surprise Trên `subset_v0.csv`

Ngày chạy: 2026-06-14

Mục tiêu:

- Kiểm tra nhanh giả thuyết LeWM-SDN: lỗi dự đoán latent kế tiếp có thể dùng làm
  anomaly score hay không.
- Chưa dùng PyTorch; prototype chỉ dùng `numpy` để chạy chắc trên môi trường nhẹ.

Mô hình:

- Encoder: random projection cố định từ feature bảng sang latent 32 chiều.
- Predictor: ridge-regression tuyến tính, train trên các cặp benign liên tiếp.
- Score: mean squared error giữa latent dự đoán và latent thật ở bước kế tiếp.

Lệnh:

```powershell
python tools\lewm_sdn_v1_numpy.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Thông tin chạy ban đầu:

- Rows: 15,000.
- Numeric features: 82.
- Latent dim: 32.
- Train pairs: 7,376.
- Train benign pairs: 2,460.
- Test pairs: 1,377.
- Threshold mặc định: quantile 0.95 trên train benign prediction error.

Kết quả:

| Thiết lập | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold benign q=0.95 | 0.3326 | 0.4630 | 0.0273 | 0.0516 | 0.7886 | 0.7844 |
| best-F1 diagnostic trên test | 0.8598 | 0.8361 | 0.9814 | 0.9030 | 0.7886 | 0.7844 |

Ghi chú:

- AUROC/AUPRC khoảng 0.79 cho thấy prediction error có tín hiệu phân biệt attack
  và benign, nhưng chưa mạnh bằng baseline supervised/toy baseline.
- F1 rất thấp với threshold q=0.95 vì ngưỡng này quá bảo thủ, dẫn đến recall chỉ
  0.0273.
- `best-F1 diagnostic trên test` chỉ dùng để hiểu khả năng của score nếu chọn
  ngưỡng tốt; không được xem là quy trình triển khai thật vì đã dùng test để
  chọn ngưỡng.
- `subset_v0.csv` không bảo toàn đầy đủ thứ tự thời gian gốc, nên V1 hiện chỉ là
  prototype. Bước tiếp theo nên tạo subset/loader giữ thứ tự dòng tốt hơn và
  calibration threshold bằng validation set.

## V1 Prototype: Validation-Calibrated Threshold

Ngày chạy: 2026-06-14

Thay đổi so với lần chạy trước:

- Chia dữ liệu thành train/validation/test.
- Predictor vẫn train trên các cặp benign trong train split.
- Ngưỡng `prediction error` được chọn bằng best-F1 trên validation.
- Test chỉ dùng để báo cáo kết quả cuối.

Lệnh:

```powershell
python tools\lewm_sdn_v1_numpy.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Thông tin chạy:

- Rows: 15,000.
- Numeric features: 82.
- Latent dim: 32.
- Train pairs: 5,444.
- Train benign pairs: 1,811.
- Validation pairs: 627.
- Test pairs: 592.
- Train benign threshold q=0.95: 0.603254.
- Validation best-F1 threshold: 0.249226.

Kết quả test:

| Thiết lập | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train benign q=0.95 | 0.3530 | 0.5135 | 0.0495 | 0.0903 | 0.8257 | 0.7953 |
| validation-calibrated | 0.8649 | 0.8439 | 0.9714 | 0.9031 | 0.8257 | 0.7953 |

Ghi chú:

- Đây là kết quả V1 hợp lệ hơn vì threshold được chọn trên validation, không chọn
  trực tiếp trên test.
- Prediction error có tín hiệu phân biệt rõ hơn lần chạy trước: AUROC test đạt
  0.8257.
- Ngưỡng q=0.95 từ train benign vẫn quá bảo thủ. Điều này cho thấy V1 cần bước
  calibration threshold nếu dùng trong pipeline detection.
- Vẫn chưa phải kết luận cuối vì `subset_v0.csv` là subset debug, chưa phải
  chuỗi traffic đầy đủ theo thời gian.
