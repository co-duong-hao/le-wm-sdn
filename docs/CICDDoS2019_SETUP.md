# Thiết Lập Dataset CICDDoS2019

Dự án này bắt đầu với bản `CSV` của CICDDoS2019. Không đưa dataset vào git vì
file dữ liệu rất nặng và không phù hợp để lưu trong mã nguồn.

## Tải Dataset

1. Mở trang dataset chính thức:
   <https://www.unb.ca/cic/datasets/ddos-2019.html>
2. Chọn thư mục `CSVs`.
3. Tải các file hoặc archive CSV về máy.
4. Giải nén vào một thư mục nằm ngoài git, ví dụ:

```text
C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs
```

Ở phiên bản đầu, chưa cần tải thư mục `PCAPs`. PCAP rất nặng và chỉ cần thiết
khi ta muốn tự trích xuất lại flow bằng CICFlowMeter.

## Kiểm Tra Ban Đầu

Sau khi giải nén, quay lại thư mục repo trước:

```powershell
cd "C:\Users\ADMIN\OneDrive\Desktop\New folder (2)\le-wm-sdn"
```

Sau đó chạy:

```powershell
python tools\inspect_cicddos2019.py --root "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs"
```

Nếu thư mục lớn và muốn kiểm tra nhanh trước:

```powershell
python tools\inspect_cicddos2019.py --root "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs" --max-files 2 --chunksize 50000
```

Nếu bạn đang đứng sẵn trong thư mục dataset `...\CICDDoS2019\CSVs`, có thể chạy
bằng đường dẫn tuyệt đối tới script:

```powershell
python "C:\Users\ADMIN\OneDrive\Desktop\New folder (2)\le-wm-sdn\tools\inspect_cicddos2019.py" --root "." --max-files 2 --chunksize 50000
```

Script sẽ báo cáo:

- Số file CSV tìm thấy.
- Danh sách cột và cột nhãn có khả năng là label.
- Số dòng trong từng file.
- Phân bố nhãn.
- Số lượng feature dạng số.

## Tạo Subset Nhỏ Để Thử Nghiệm

CICDDoS2019 rất lớn và lệch lớp mạnh. Không nên train trực tiếp trên toàn bộ dữ
liệu ở bước đầu. Sau khi kiểm tra schema, tạo một file subset nhỏ:

```powershell
python tools\make_cicddos2019_subset.py --root "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs" --out "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv" --max-files 2 --max-per-label 5000 --chunksize 50000
```

Ý nghĩa:

- `--max-files 2`: chỉ dùng 2 file đầu để chạy thử nhanh.
- `--max-per-label 5000`: tối đa 5000 dòng cho mỗi nhãn.
- `--chunksize 50000`: đọc từng phần nhỏ, tránh load toàn bộ CSV vào RAM.
- `--out`: nơi lưu subset, nằm ngoài repo.

Khi lệnh thử chạy ổn, có thể bỏ `--max-files 2` để tạo subset từ toàn bộ dataset.

## Tạo Subset Giữ Thứ Tự Cho V1

Với LeWM-SDN V1, ta cần các cặp liên tiếp để học dự đoán latent kế tiếp. Vì vậy
có thêm subset giữ thứ tự dòng trong từng nhãn:

```powershell
python tools\make_cicddos2019_ordered_subset.py --root "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs" --out "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v1_ordered.csv" --max-files 2 --max-per-label 5000 --chunksize 50000
```

Khác biệt chính:

- `make_cicddos2019_subset.py`: lấy mẫu ngẫu nhiên theo nhãn, tốt cho debug.
- `make_cicddos2019_ordered_subset.py`: lấy first-N theo thứ tự xuất hiện, tốt
  hơn cho thí nghiệm world-model ban đầu.

## Chạy Baseline V0

Sau khi có `subset_v0.csv`, chạy baseline nhẹ:

```powershell
python tools\baseline_cicddos2019.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Baseline này chưa phải LeWM-SDN. Nó chỉ là mốc tham chiếu ban đầu để kiểm tra
pipeline:

- `benign-zscore-anomaly`: học thống kê từ traffic benign rồi chấm điểm bất
  thường bằng khoảng cách z-score.
- `nearest-centroid-binary`: classifier nhị phân rất đơn giản giữa benign và
  attack.

Các metric cần ghi lại: accuracy, precision, recall, F1, AUROC, AUPRC.

## Chạy LeWM-SDN V1 Prototype

Prototype V1 kiểm tra tín hiệu `prediction surprise` trong latent space bằng
một mô hình nhẹ chỉ dùng `numpy`:

```powershell
python tools\lewm_sdn_v1_numpy.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Với subset ordered, chạy:

```powershell
python tools\lewm_sdn_v1_numpy.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v1_ordered.csv" --split-mode block
```

Lưu ý: `subset_v0.csv` phù hợp để kiểm tra pipeline nhưng chưa phải tập chuỗi
thời gian lý tưởng. Kết quả V1 trên file này chỉ là mốc prototype, chưa dùng làm
kết luận cuối.

Trong output, ưu tiên đọc dòng:

```text
[lewm-sdn-v1-validation-calibrated-test]
```

Dòng này dùng threshold được chọn trên validation set, rồi báo cáo kết quả trên
test set.

## Quy Ước Dataset Cho Version 1

Với phiên bản LeWM-SDN detection đầu tiên:

- Dùng feature flow từ CSV, không dùng PCAP.
- Chỉ dùng label để đánh giá và so sánh baseline, không dùng label làm tín hiệu
  chính cho mô hình LeWM-style tự giám sát.
- Huấn luyện predictor LeWM-style chủ yếu trên các cửa sổ traffic benign.
- Dùng lỗi dự đoán latent kế tiếp làm điểm bất thường.
- Luôn có chế độ subset nhỏ để phù hợp RTX 3050 Laptop 4GB VRAM.
