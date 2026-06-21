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

## 9. Quy Trình Retrain Tự Động (Lấy dữ liệu từ Web)

Hệ thống cung cấp một luồng (pipeline) hoàn chỉnh để tự động xuất dữ liệu (export) các claim mới nhất từ cơ sở dữ liệu MySQL của project Web sang hệ thống ML, sau đó tiến hành retrain tự động.

### Chạy tự động toàn bộ (Khuyến nghị)
Bạn chỉ cần chạy file `.bat` đã được cấu hình sẵn ở thư mục gốc:

```bash
.\retrain_with_export.bat
```

**Script này thực hiện 2 bước:**
1. **Xuất Dữ Liệu:** Gọi `src\export_claims_csv.py` để kết nối vào database MySQL, lấy các claim mới và lưu thành file `data/new/claims_YYYYMMDD.csv`.
2. **Retrain:** Gọi `src\retrain_cycle.py` để gộp data cũ và mới, loại bỏ trùng lặp, chạy lại toàn bộ quy trình auto-selection và chọn ra model tốt nhất.

**Các tùy chọn chạy nâng cao:**
```bash
retrain_with_export.bat --since 7d          # Lấy dữ liệu 7 ngày qua
retrain_with_export.bat --since 2026-06-01  # Lấy từ ngày cụ thể
```

### Thiết lập chạy tự động định kỳ (Cronjob/Task Scheduler)
Bạn có thể cài đặt `Windows Task Scheduler` để tự động retrain hàng ngày/tuần:
- **Program/script:** `E:\SCHOOL\PTL\retrain_with_export.bat`
- **Start in:** `E:\SCHOOL\PTL`

### Chạy thủ công từng bước (Manual)
Nếu bạn không dùng MySQL mà có sẵn file CSV mới, chỉ cần thả các file CSV đó vào thư mục `data/new/` và chạy lệnh sau (retrain với data của 3 ngày gần nhất):

```bash
python src\retrain_cycle.py --days 3 --label-mode synthetic
```

Kết quả retrain luôn được tự động lưu theo thời gian tại:
```text
outputs/retrain/<timestamp>/  (Lưu ảnh ROC, biểu đồ, JSON report...)
models/retrain/<timestamp>/   (Lưu model .joblib)
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

## 12. Chạy API Server (Phục vụ Frontend)

Hệ thống cung cấp một FastAPI backend để frontend có thể gọi suy luận (predict) và lấy dữ liệu hiển thị.

Lệnh khởi động server:

```bash
python -m uvicorn src.app.api:app --reload --port 8000
```

Các endpoint API chính:
- `POST /predict`: Dự đoán gian lận cho 1 hồ sơ yêu cầu bồi thường (claim).
- `POST /predict_batch`: Dự đoán cho nhiều claim cùng lúc.
- `GET /model/status`: Kiểm tra trạng thái model hiện hành.
- `GET /model/history`: Trả về danh sách lịch sử retrain và URL của các biểu đồ đánh giá.
- `POST /explain`: Giải thích quyết định của model sử dụng SHAP.

*Lưu ý:* Các file hình ảnh đánh giá (ROC, PR Curve...) được phục vụ tĩnh tự động qua đường dẫn `/outputs/` (ví dụ: `http://localhost:8000/outputs/retrain/20260621_155802/roc_curves.png`).

## 13. Lệnh Test Khuyến Nghị

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
