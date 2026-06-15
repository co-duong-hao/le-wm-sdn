# LeWM-SDN Demo

Branch này là bản demo nghiên cứu ban đầu cho hướng áp dụng ý tưởng
LeWorldModel vào bài toán phát hiện DDoS trên dữ liệu network flow.

Demo hiện tại chỉ dùng các script trong `tools/`. Phần code LeWM gốc chưa được
tích hợp vào pipeline train SDN chính thức.

## Mục tiêu demo

Demo chứng minh được các bước:

1. Đọc được dataset CICDDoS2019 bản CSV.
2. Tạo được subset nhỏ, cân bằng để debug pipeline.
3. Chạy được baseline V0 làm mốc tham chiếu.
4. Chạy được prototype LeWM-SDN V1 dùng lỗi dự đoán latent kế tiếp làm anomaly
   score.

Đây chưa phải sản phẩm cuối và chưa kết luận LeWM-SDN tốt hơn ML/DL truyền
thống. Đây là mốc V0/V1 để kiểm tra pipeline và tín hiệu `prediction surprise`.

## Yêu cầu môi trường

Các script demo chỉ cần:

- Python 3.10+.
- `numpy`.
- `pandas`.

Không cần `torch`, `stable-worldmodel` hoặc `scikit-learn` để chạy demo V0/V1.

Kiểm tra nhanh:

```powershell
python -c "import numpy, pandas; print('ok')"
```

## Chuẩn bị dataset

Tải CICDDoS2019 từ trang chính thức:

<https://www.unb.ca/cic/datasets/ddos-2019.html>

Chọn thư mục `CSVs`, tải và giải nén ra ngoài repo, ví dụ:

```text
C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs
```

Không đặt dataset vào git. Repo đã ignore:

```text
datasets/
data/
outputs/
runs/
checkpoints/
*.pt
*.pth
*.ckpt
```

## 1. Kiểm tra dataset

Chạy từ thư mục repo:

```powershell
cd "C:\Users\ADMIN\OneDrive\Desktop\New folder (2)\le-wm-sdn"

python tools\inspect_cicddos2019.py --root "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs" --max-files 2 --chunksize 50000
```

Output cần chú ý:

- `rows`: số flow trong mỗi file.
- `columns`: số cột gốc.
- `label column`: cột nhãn.
- `top labels`: phân bố nhãn.
- `numeric feature columns`: số feature dạng số dùng được cho model.

## 2. Tạo subset V0

Dataset gốc rất lớn và lệch lớp mạnh, nên demo dùng subset nhỏ:

```powershell
python tools\make_cicddos2019_subset.py --root "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\CSVs" --out "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv" --max-files 2 --max-per-label 5000 --chunksize 50000
```

Subset V0 đã dùng trong thí nghiệm:

```text
15,000 dòng
82 numeric features
BENIGN: 5,000
DrDoS_DNS: 5,000
DrDoS_LDAP: 5,000
```

## 3. Chạy baseline V0

```powershell
python tools\baseline_cicddos2019.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Kết quả tham chiếu đã ghi nhận:

```text
[benign-zscore-anomaly]
f1    : 0.9169
auroc : 0.9862

[nearest-centroid-binary]
f1    : 0.9919
auroc : 0.9950
```

Baseline này dùng để kiểm tra pipeline và làm mốc so sánh. Vì subset hiện chỉ có
2 loại attack và đã được cân bằng, kết quả baseline cao là hợp lý.

## 4. Chạy LeWM-SDN V1 prototype

```powershell
python tools\lewm_sdn_v1_numpy.py --csv "C:\Users\ADMIN\OneDrive\Desktop\datasets\CICDDoS2019\subset_v0.csv"
```

Prototype V1 hiện tại:

```text
flow numeric features
-> random latent projection
-> linear next-latent predictor trained on benign pairs
-> prediction error as anomaly score
-> validation threshold calibration
-> test metrics
```

Dòng kết quả quan trọng trong output:

```text
[lewm-sdn-v1-validation-calibrated-test]
accuracy : 0.8649
precision: 0.8439
recall   : 0.9714
f1       : 0.9031
auroc    : 0.8257
auprc    : 0.7953
```

Cách diễn giải:

- `prediction error` trong latent space đã có tín hiệu phát hiện attack.
- V1 hiện chưa mạnh bằng baseline đơn giản trên subset này.
- Bước tiếp theo là tạo subset/loader giữ thứ tự thời gian tốt hơn và thay
  random projection/linear predictor bằng encoder-predictor học được.

## Tài liệu liên quan

- Roadmap nghiên cứu: [`docs/LEWM_SDN_ROADMAP.md`](docs/LEWM_SDN_ROADMAP.md)
- Hướng dẫn thiết lập CICDDoS2019: [`docs/CICDDoS2019_SETUP.md`](docs/CICDDoS2019_SETUP.md)
- Cách đọc output và metric: [`docs/HOW_TO_READ_OUTPUTS.md`](docs/HOW_TO_READ_OUTPUTS.md)
- Nhật ký thí nghiệm: [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)

