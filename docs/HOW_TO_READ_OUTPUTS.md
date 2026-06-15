# Cách Đọc Output Và Metric

Tài liệu này dùng để đọc nhanh các kết quả khi làm việc với CICDDoS2019 và các
baseline/LeWM-SDN prototype.

## 1. Output Của `inspect_cicddos2019.py`

Ví dụ:

```text
[file] DrDoS_DNS.csv
  rows: 5,074,413
  columns: 88
  label column:  Label
  top labels:
    DrDoS_DNS: 5,071,011
    BENIGN: 3,402

[summary]
total rows: 7,255,955
numeric feature columns: 83
```

Cách đọc:

- `rows`: số flow trong file CSV.
- `columns`: tổng số cột gốc.
- `label column`: cột nhãn. CICDDoS2019 thường có tên ` Label` với dấu cách
  đầu, nên script sẽ tự chuẩn hóa.
- `top labels`: các nhãn xuất hiện nhiều nhất trong file.
- `numeric feature columns`: số feature dạng số có thể dùng cho mô hình.

Cần chú ý:

- Nếu `BENIGN` quá ít so với attack, dataset đang lệch lớp mạnh.
- Nếu không tìm thấy label column, không nên train tiếp.
- Nếu số numeric feature quá thấp, có thể parser đọc lỗi hoặc file không đúng.

## 2. Output Của `make_cicddos2019_subset.py`

Ví dụ:

```text
[seen labels]
  DrDoS_DNS: 5,071,011
  DrDoS_LDAP: 2,179,930
  BENIGN: 5,014

[written labels]
  DrDoS_DNS: 5,000
  BENIGN: 5,000
  DrDoS_LDAP: 5,000
```

Cách đọc:

- `seen labels`: nhãn thật sự thấy trong dữ liệu gốc khi stream CSV.
- `written labels`: nhãn đã được ghi vào file subset.

Cần chú ý:

- `seen labels` cho biết dữ liệu gốc lệch lớp thế nào.
- `written labels` cho biết subset có cân bằng hay không.
- Nếu một nhãn ở `seen labels` nhưng không có trong `written labels`, có thể
  `--max-files` đang giới hạn quá ít file hoặc nhãn đó không đủ mẫu.

## 3. Output Của Baseline

Ví dụ:

```text
[benign-zscore-anomaly]
accuracy : 0.8947
precision: 0.9667
recall   : 0.8720
f1       : 0.9169
auroc    : 0.9862
auprc    : 0.9909
```

Cách đọc metric:

- `accuracy`: tỷ lệ dự đoán đúng tổng thể. Không nên dựa quá nhiều vào chỉ số
  này nếu dataset lệch lớp.
- `precision`: trong các mẫu bị báo là attack, bao nhiêu mẫu thật sự là attack.
  Precision thấp nghĩa là báo động giả nhiều.
- `recall`: trong toàn bộ attack thật, mô hình bắt được bao nhiêu. Recall thấp
  nghĩa là bỏ lọt attack nhiều.
- `f1`: trung bình điều hòa giữa precision và recall. Đây là metric gọn, hữu ích
  khi cần một con số để so sánh.
- `auroc`: khả năng tách benign và attack trên nhiều ngưỡng. Gần 1 là tốt.
- `auprc`: quan trọng khi dữ liệu lệch lớp; thể hiện quan hệ precision/recall ở
  nhiều ngưỡng.

Trong bài toán DDoS detection:

- `recall` cao giúp giảm bỏ lọt attack.
- `precision` cao giúp giảm chặn nhầm traffic bình thường.
- `AUROC/AUPRC` cao cho thấy score có khả năng xếp attack cao hơn benign.

## 4. Output Của LeWM-SDN V1 Prototype

LeWM-SDN V1 dùng ý tưởng:

```text
flow features -> latent embedding -> dự đoán latent kế tiếp -> prediction error
```

Các dòng cần đọc:

- `Rows`: số dòng dùng trong thí nghiệm.
- `Numeric features`: số feature đầu vào.
- `Latent dim`: số chiều latent.
- `Train benign windows`: số cửa sổ benign dùng để học động lực học bình thường.
- `Train benign threshold`: ngưỡng anomaly score lấy từ train benign.
- `Validation best-F1 threshold`: ngưỡng được chọn trên validation set.
- `F1`, `AUROC`, `AUPRC`: chất lượng phát hiện bằng prediction error.
- `validation-calibrated-test`: kết quả test khi ngưỡng đã được chọn bằng
  validation. Đây là dòng quan trọng hơn dòng diagnostic.

Cần nhớ:

- Nếu V1 chạy trên `subset_v0.csv`, đây mới là prototype vì subset này không bảo
  toàn đầy đủ thứ tự thời gian gốc.
- Kết quả V1 chính thức cần subset/loader có giữ thứ tự thời gian hoặc trace SDN
  rõ ràng hơn.

## 5. DNS Và LDAP Trong Dataset

`DNS` là Domain Name System, hệ thống phân giải tên miền sang địa chỉ IP.

`LDAP` là Lightweight Directory Access Protocol, giao thức truy vấn thư mục/tài
khoản/quyền truy cập trong hệ thống doanh nghiệp.

`DrDoS` là Distributed Reflection Denial of Service, tức DDoS phản xạ/khuếch đại.
Attacker giả mạo IP nạn nhân, gửi request tới server trung gian như DNS/LDAP,
rồi server trung gian trả response lớn về nạn nhân.

Vì vậy:

- `DrDoS_DNS`: DDoS phản xạ/khuếch đại qua DNS.
- `DrDoS_LDAP`: DDoS phản xạ/khuếch đại qua LDAP.
