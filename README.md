# CMS Fraud Anomaly Detection Project

Project này xây dựng pipeline phát hiện bất thường/gian lận trong hồ sơ yêu cầu bồi thường bảo hiểm y tế Medicare, sử dụng dữ liệu **CMS Synthetic Inpatient Claims**.

Mục tiêu chính:

- Tiền xử lý dữ liệu claims.
- Tạo feature nghiệp vụ từ chi phí, thời gian nằm viện, mã chẩn đoán/thủ thuật, provider/physician.
- Chạy và so sánh 4 mô hình anomaly detection.
- Tự chọn mô hình phù hợp với dữ liệu.
- Xuất danh sách claim nghi ngờ để phục vụ điều tra.
- Hỗ trợ retrain định kỳ khi có dữ liệu mới.

## 1. Mô Hình Sử Dụng

Project hiện hỗ trợ 4 mô hình:

1. **Isolation Forest**
   - Phù hợp làm baseline mạnh.
   - Tốt với dữ liệu lớn, nhiều feature.

2. **CBLOF**
   - Phát hiện điểm/cụm nhỏ nằm xa cụm lớn.
   - Trong lần chạy full dataset hiện tại, CBLOF đang là model tốt nhất theo F1 và ROC-AUC.

3. **One-Class SVM**
   - Học ranh giới vùng dữ liệu bình thường.
   - Phù hợp hơn với dataset nhỏ/vừa, không quá nhiều feature.

4. **ECOD**
   - Phát hiện điểm nằm ở vùng đuôi phân phối.
   - Chạy nhanh, phù hợp dữ liệu numeric-heavy.

## 2. Cấu Trúc Thư Mục

```text
cms_fraud_anomaly_project/
├── data/
│   ├── DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
│   └── new/                         # đặt data mới tại đây nếu cần retrain
├── models/                          # model đã train
├── outputs/                         # kết quả đánh giá, score, biểu đồ
├── src/
│   ├── features.py                  # feature engineering
│   ├── labels.py                    # tạo synthetic anomaly label
│   ├── models.py                    # 4 mô hình anomaly detection
│   ├── evaluation.py                # metric và biểu đồ
│   ├── model_selection.py           # tự chọn model phù hợp
│   ├── retrain_cycle.py             # retrain với data mới theo chu kỳ
│   ├── run_experiment.py            # script chạy chính
│   └── tuning.py                    # tuning tham số
├── requirements.txt
└── README.md
```

## 3. Cài Đặt

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

## 4. Chạy Test Nhanh

Dùng lệnh này để test pipeline với 3.000 dòng dữ liệu:

```bash
python src\run_experiment.py --data data\DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 3000 --model-mode auto --output-dir outputs\test_auto --model-dir models\test_auto
```

Lệnh trên sẽ:

- Đọc dữ liệu.
- Tạo feature nghiệp vụ.
- Tạo synthetic anomaly label để đánh giá.
- Tự phân tích dữ liệu xem model nào phù hợp.
- Vì có label đánh giá, hệ thống sẽ chạy cả 4 model.
- Chọn model tốt nhất theo F1, ROC-AUC và Average Precision.
- Lưu kết quả vào `outputs/test_auto/`.

Sau khi chạy, xem các file:

```text
outputs/test_auto/model_suitability.csv
outputs/test_auto/model_selection_report.json
outputs/test_auto/model_comparison.csv
outputs/test_auto/top_suspicious_claims.csv
```

## 5. Chạy Full Dataset

Chạy toàn bộ dữ liệu:

```bash
python src\run_experiment.py --data data\DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 0 --model-mode auto
```

Kết quả chính được lưu tại:

```text
outputs/model_suitability.csv
outputs/model_selection_report.json
outputs/model_comparison.csv
outputs/claim_anomaly_scores.csv
outputs/top_suspicious_claims.csv
outputs/roc_curves.png
outputs/pr_curves.png
outputs/confusion_matrices.png
outputs/score_distributions.png
```

Model được lưu tại:

```text
models/preprocessor.joblib
models/IsolationForest.joblib
models/CBLOF.joblib
models/OCSVM.joblib
models/ECOD.joblib
```

## 6. Ý Nghĩa `label-mode`

Project hỗ trợ 3 chế độ label:

```text
--label-mode synthetic
```

Tạo nhãn bất thường mô phỏng từ rule nghiệp vụ. Chế độ này dùng để đánh giá trong điều kiện không có nhãn gian lận thật.

```text
--label-mode column --label-col ten_cot_nhan
```

Dùng cột nhãn có sẵn trong dataset.

```text
--label-mode none
```

Không dùng nhãn. Chỉ xuất anomaly score, không tính Precision/Recall/F1/ROC-AUC.

Ví dụ dùng cột nhãn thật nếu có:

```bash
python src\run_experiment.py --data data\claims.csv --label-mode column --label-col anomaly_label --model-mode auto
```

## 7. Tự Chọn Model Phù Hợp

Tham số `--model-mode` quyết định cách chạy model:

```text
--model-mode all
```

Chạy cả 4 model. Đây là chế độ mặc định.

```text
--model-mode recommended
```

Chạy 2 model phù hợp nhất theo đặc điểm dữ liệu.

```text
--model-mode auto
```

Chế độ khuyến nghị.

- Nếu có label đánh giá: chạy cả 4 model rồi chọn model tốt nhất theo metric.
- Nếu không có label: chọn model phù hợp nhất theo heuristic dữ liệu.

Heuristic dựa trên:

- Số dòng dữ liệu.
- Số lượng feature.
- Tỷ lệ numeric/categorical.
- Tỷ lệ missing value.
- Tỷ lệ anomaly kỳ vọng (`contamination`).

## 8. Đọc Kết Quả Chọn Model

File `outputs/model_suitability.csv` cho biết model nào phù hợp với đặc điểm dữ liệu trước khi đánh giá metric.

Ví dụ:

```text
rank,model,suitability_score,reason
1,ECOD,86.0,...
2,IsolationForest,81.0,...
3,CBLOF,64.0,...
4,OCSVM,31.0,...
```

File `outputs/model_selection_report.json` cho biết:

- Profile của dataset.
- Model được heuristic gợi ý.
- Các model đã chạy.
- Model cuối cùng được chọn theo metric nếu có label.

Trong lần chạy full dataset hiện tại, hệ thống chọn:

```text
Selected model: CBLOF
F1-score: 0.4582
ROC-AUC: 0.9059
```

## 9. Retrain Với Data Mới Sau 3 Ngày

Nếu có dữ liệu mới thu thập được, đặt các file CSV vào:

```text
data/new/
```

Chạy retrain với các file mới trong 3 ngày gần nhất:

```bash
python src\retrain_cycle.py --days 3 --label-mode synthetic
```

Script sẽ:

1. Đọc data nền hiện tại.
2. Tìm file CSV mới trong `data/new/`.
3. Gộp data cũ và data mới.
4. Drop duplicate theo `CLM_ID` nếu có.
5. Chạy lại experiment với `--model-mode auto`.
6. Chọn lại model phù hợp.
7. Lưu kết quả theo timestamp.

Kết quả retrain nằm tại:

```text
outputs/retrain/<timestamp>/
models/retrain/<timestamp>/
```

Nếu muốn bắt buộc phải có data mới thì dùng:

```bash
python src\retrain_cycle.py --days 3 --require-new-data
```

## 10. Chạy Tuning Tham Số

Nếu muốn tuning tham số cho 4 model:

```bash
python src\run_experiment.py --data data\DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 30000 --model-mode all --tune
```

Kết quả tuning:

```text
outputs/tuning_results.csv
outputs/best_tuning_results.csv
```

## 11. Lưu Ý Khi Viết Báo Cáo

Dữ liệu CMS Synthetic thường không có nhãn gian lận thật. Vì vậy:

- `synthetic_anomaly_label` chỉ là nhãn mô phỏng.
- Kết quả Precision/Recall/F1/ROC-AUC là đánh giá trong điều kiện mô phỏng.
- Khi viết khóa luận, nên trình bày rõ đây là **synthetic anomaly evaluation** hoặc **controlled stress test**.

Không nên nói hệ thống đã phát hiện “gian lận thật” nếu chưa có nhãn fraud thực tế được xác minh bởi chuyên gia.

## 12. Lệnh Test Khuyến Nghị

Test nhanh:

```bash
python src\run_experiment.py --data data\DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 3000 --model-mode auto --output-dir outputs\test_auto --model-dir models\test_auto
```

Chạy thật:

```bash
python src\run_experiment.py --data data\DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 0 --model-mode auto
```

Retrain với data mới:

```bash
python src\retrain_cycle.py --days 3 --label-mode synthetic
```
