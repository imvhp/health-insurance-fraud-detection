# 3.6. Pipeline Machine Learning

## Tổng quan

Pipeline học máy được xây dựng nhằm phát hiện bất thường trong hồ sơ yêu cầu bồi thường bảo hiểm y tế nội trú Medicare, được triển khai theo kiến trúc mô-đun hóa gồm sáu giai đoạn tuần tự: thu thập dữ liệu, tiền xử lý và khám phá, trích xuất đặc trưng, huấn luyện mô hình, đánh giá, và tổng kết. Bốn thuật toán phát hiện bất thường không giám sát được so sánh trực tiếp trong cùng điều kiện thực nghiệm, kết hợp với cơ chế lựa chọn mô hình tự động hai tầng dựa trên đặc điểm dữ liệu và chỉ số đánh giá thực nghiệm.

---

## Phần 1: Thiết lập và Thu thập Dữ liệu

### 1.1. Nguồn Dữ liệu

Nghiên cứu sử dụng bộ dữ liệu **CMS DE-SynPUF — Data Entrepreneurs' Synthetic Public Use File**, cụ thể là tập hồ sơ yêu cầu bồi thường nội trú (*Inpatient Claims*) mẫu số 1, bao gồm các giao dịch phát sinh trong giai đoạn 2008–2010. Bộ dữ liệu được Trung tâm Dịch vụ Medicare và Medicaid Hoa Kỳ (CMS) công bố dưới dạng tổng hợp (synthetic), nghĩa là thông tin bệnh nhân và nhà cung cấp đã được mã hóa hoặc sửa đổi để bảo vệ quyền riêng tư. Hệ quả quan trọng là bộ dữ liệu **không chứa nhãn gian lận thực tế được xác minh** — đây là hạn chế cơ bản cần được phản ánh rõ trong mọi diễn giải kết quả.

Dữ liệu bao gồm các nhóm trường thông tin sau:

| Nhóm | Các trường tiêu biểu | Ý nghĩa |
|---|---|---|
| Định danh | `DESYNPUF_ID`, `CLM_ID`, `SEGMENT` | Mã bệnh nhân, mã claim, mã phân đoạn |
| Thời gian | `CLM_FROM_DT`, `CLM_THRU_DT`, `CLM_ADMSN_DT`, `NCH_BENE_DSCHRG_DT` | Mốc thời gian claim và nhập/xuất viện (định dạng YYYYMMDD) |
| Tài chính | `CLM_PMT_AMT`, `NCH_PRMRY_PYR_CLM_PD_AMT`, `NCH_BENE_IP_DDCTBL_AMT`, `CLM_PASS_THRU_PER_DIEM_AMT` | Số tiền bồi thường, phần thanh toán của bảo hiểm chính, khấu trừ, phần mỗi ngày |
| Sử dụng | `CLM_UTLZTN_DAY_CNT` | Số ngày sử dụng dịch vụ |
| Nhà cung cấp / Bác sĩ | `PRVDR_NUM`, `AT_PHYSN_NPI`, `OP_PHYSN_NPI`, `OT_PHYSN_NPI` | Mã provider, mã NPI bác sĩ điều trị, phẫu thuật, bác sĩ khác |
| Mã y tế | `ICD9_DGNS_CD_1` … `_10`, `ICD9_PRCDR_CD_1` … `_6`, `CLM_DRG_CD`, `ADMTNG_ICD9_DGNS_CD` | Mã chẩn đoán ICD-9 (tối đa 10 mã), mã thủ thuật (tối đa 6), mã DRG |

### 1.2. Chiến lược Nạp và Lấy Mẫu Dữ liệu

Dữ liệu được nạp trực tiếp từ file CSV. Trong giai đoạn phát triển và kiểm thử, hệ thống hỗ trợ lấy mẫu ngẫu nhiên không hoàn lại (*sampling without replacement*) với kích thước mẫu có thể cấu hình — mặc định là 3.000 hồ sơ cho kiểm thử nhanh và toàn bộ tập dữ liệu khi chạy thực nghiệm đầy đủ. Tham số `random_state = 42` được cố định để đảm bảo tính tái lập (*reproducibility*). Kết quả được phân tích trong báo cáo này dựa trên tập 3.000 hồ sơ.

### 1.3. Cơ chế Cập nhật Định kỳ

Hệ thống được thiết kế để hỗ trợ chu kỳ retrain khi có dữ liệu claim mới. Dữ liệu bổ sung được gộp với dữ liệu nền lịch sử, loại bỏ bản ghi trùng lặp theo khóa chính `CLM_ID` (hoặc theo tất cả cột nếu `CLM_ID` không tồn tại), sau đó chạy lại toàn bộ quy trình huấn luyện và lựa chọn mô hình. Mỗi chu kỳ retrain được lưu vào thư mục riêng đánh dấu theo timestamp, đảm bảo khả năng truy vết và so sánh lịch sử phiên bản mô hình.

---

## Phần 2: Tiền xử lý và Khám phá Dữ liệu

### 2.1. Xây dựng Nhãn Bất thường Mô phỏng

Do bộ dữ liệu CMS không có nhãn gian lận thực tế, nghiên cứu áp dụng phương pháp **synthetic anomaly labeling**: xây dựng nhãn bất thường mô phỏng dựa trên tổ hợp các quy tắc nghiệp vụ có trọng số, xuất phát từ các chỉ dấu gian lận bảo hiểm y tế phổ biến được ghi nhận trong tài liệu chuyên ngành.

Với mỗi hồ sơ, điểm bất thường tổng hợp $s_i$ được tính theo công thức cộng dồn:

$$s_i = \sum_{k} w_k \cdot \mathbf{1}[\text{điều kiện}_k(x_i)]$$

trong đó $w_k$ là trọng số của quy tắc thứ $k$ và $\mathbf{1}[\cdot]$ là hàm chỉ thị. Các quy tắc và trọng số cụ thể được trình bày trong Bảng 2.1.

**Bảng 2.1.** Quy tắc và trọng số xây dựng nhãn bất thường mô phỏng.

| Chỉ dấu bất thường | Điều kiện kích hoạt | Trọng số $w_k$ |
|---|---|:---:|
| Chi phí claim rất cao | `CLM_PMT_AMT` > phân vị 95% | 1,0 |
| Chi phí claim cực cao | `CLM_PMT_AMT` > phân vị 99% | +2,0 (cộng dồn) |
| Thời gian điều trị kéo dài | `CLM_UTLZTN_DAY_CNT`, `claim_duration_days`, hoặc `admission_duration_days` > phân vị 95% | 0,8 |
| Thời gian điều trị quá dài | Các biến trên > phân vị 99% | +1,5 (cộng dồn) |
| Chi phí mỗi ngày bất thường | `amount_per_utilization_day` > phân vị 97% | 1,5 |
| Upcoding mã chẩn đoán/thủ thuật | `num_diagnosis_codes` hoặc `num_procedure_codes` > phân vị 95% | 0,7 |
| Provider/physician hiếm gặp kèm chi phí cao | Tần suất xuất hiện ≤ phân vị 5% **và** `CLM_PMT_AMT` > phân vị 90% | 0,5 |

Sau khi tính $s_i$ cho toàn bộ $N$ hồ sơ, nhãn nhị phân $y_i \in \{0, 1\}$ được gán cho $\lfloor \tau \times N \rfloor$ hồ sơ có điểm tổng hợp cao nhất, trong đó $\tau = 0{,}05$ là tỉ lệ nhiễm mặc định (*contamination rate*). Phương pháp này tạo ra môi trường kiểm thử có kiểm soát (*controlled stress test*) và không thay thế cho nhãn gian lận được xác minh thực tế.

> **Lưu ý về tính giá trị khoa học**: Toàn bộ chỉ số Precision, Recall, F1 và ROC-AUC trong nghiên cứu này được tính trên nhãn mô phỏng (*synthetic anomaly evaluation*). Kết quả phản ánh khả năng của mô hình trong việc tái phát hiện các chỉ dấu nghiệp vụ đã định nghĩa — không phản ánh hiệu suất trên gian lận thực tế chưa được xác minh.

### 2.2. Phân tách Tập Dữ liệu

Dữ liệu được phân tách theo tỉ lệ 70% huấn luyện / 30% kiểm tra bằng phương pháp **phân tầng** (*stratified split*) theo nhãn $y$, nhằm đảm bảo tỉ lệ bất thường được phân bố đồng đều trong cả hai tập. Tham số `random_state = 42` được sử dụng nhất quán trong toàn bộ quy trình.

### 2.3. Đặc điểm Tập Dữ liệu Thực nghiệm

Kết quả phân tích đặc điểm (*data profiling*) trên tập thực nghiệm được trình bày trong Bảng 2.2.

**Bảng 2.2.** Đặc điểm tập dữ liệu thực nghiệm (3.000 hồ sơ).

| Thuộc tính | Giá trị |
|---|---|
| Tổng số hồ sơ | 3.000 |
| Số hồ sơ huấn luyện (70%) | 2.100 |
| Số hồ sơ kiểm tra (30%) | 900 |
| Tổng số đặc trưng sau trích xuất | 94 |
| — Đặc trưng dạng số | 77 (81,9%) |
| — Đặc trưng dạng phân loại | 17 (18,1%) |
| Tỉ lệ giá trị khuyết trung bình | 54,78% |
| Tỉ lệ bất thường mô phỏng ($\tau$) | 5,0% (150/3.000) |
| Số hồ sơ bất thường trong tập test | ~45 / 900 |

Tỉ lệ giá trị khuyết cao (54,78%) phản ánh bản chất tự nhiên của dữ liệu y tế: mỗi hồ sơ chỉ điền một tập con các mã chẩn đoán và thủ thuật tùy theo tình trạng lâm sàng. Đây cũng là lý do khiến chiến lược xử lý giá trị khuyết (*imputation strategy*) trở thành một quyết định thiết kế quan trọng trong pipeline.

---

## Phần 3: Trích xuất Đặc trưng và Chuẩn bị Dữ liệu

### 3.1. Trích xuất Đặc trưng Nghiệp vụ

Từ dữ liệu thô, bốn nhóm đặc trưng nghiệp vụ được xây dựng nhằm mã hóa các chỉ dấu bất thường tiềm năng. Quy trình được thiết kế *schema-agnostic*: mỗi đặc trưng chỉ được tạo khi các cột nguồn tương ứng tồn tại trong dataset, đảm bảo khả năng tương thích với các phiên bản dữ liệu khác nhau mà không gây lỗi runtime.

#### 3.1.1. Đặc trưng Khoảng Thời gian

Các cột ngày (định dạng YYYYMMDD) được xác định tự động thông qua biểu thức chính quy khớp với hậu tố `_DT`, `DATE`, `FROM`, `THRU`, `ADMSN`, `DSCHRG`. Ba biến khoảng thời gian được tính bằng hiệu số giữa các cặp mốc thời gian (đơn vị: ngày), được ép về giá trị không âm:

**Bảng 3.1.** Đặc trưng khoảng thời gian được dẫn xuất.

| Tên đặc trưng | Công thức | Ý nghĩa |
|---|---|---|
| `claim_duration_days` | `CLM_THRU_DT` − `CLM_FROM_DT` | Độ dài tổng thể của claim |
| `admission_duration_days` | `NCH_BENE_DSCHRG_DT` − `CLM_ADMSN_DT` | Thời gian nằm viện thực tế |
| `admission_to_claim_end_days` | `CLM_THRU_DT` − `CLM_ADMSN_DT` | Độ trễ từ nhập viện đến đóng claim |

#### 3.1.2. Đặc trưng Tài chính

| Tên đặc trưng | Công thức | Mục đích |
|---|---|---|
| `claim_amount_log1p` | $\ln(1 + \text{CLM\_PMT\_AMT})$ | Giảm độ lệch phải (*right skewness*) của phân phối chi phí |
| `amount_per_utilization_day` | $\text{CLM\_PMT\_AMT} \div \text{CLM\_UTLZTN\_DAY\_CNT}$ | Phát hiện chi phí mỗi ngày bất thường cao |

Biến đổi logarithm $\ln(1+x)$ được áp dụng thay vì $\ln(x)$ để tránh lỗi khi giá trị bằng 0. Phép chia `amount_per_utilization_day` thay thế 0 ở mẫu số bằng `NaN` để tránh vô cực (*infinity*), sau đó được xử lý trong bước imputation.

#### 3.1.3. Đặc trưng Độ Phức tạp Y tế

Các cột mã chẩn đoán được xác định bằng biểu thức chính quy khớp với `ICD.*DGNS` hoặc `DGNS`; mã thủ thuật khớp với `ICD.*PRCDR` hoặc `PRCDR`.

| Tên đặc trưng | Định nghĩa | Liên quan đến gian lận |
|---|---|---|
| `num_diagnosis_codes` | Đếm số trường `ICD9_DGNS_CD_*` không rỗng | Upcoding: khai thêm chẩn đoán để tăng mức bồi thường |
| `num_procedure_codes` | Đếm số trường `ICD9_PRCDR_CD_*` không rỗng | Khai khống thủ thuật không thực hiện |

#### 3.1.4. Đặc trưng Tần suất Provider và Bác sĩ

Với mỗi cột định danh (nhận diện qua biểu thức chính quy khớp với `ID$`, `_ID`, `NPI`, `PRVDR`, `PHYSN`, `CLM_ID`, `DESYNPUF`), hai đặc trưng tần suất được tạo ra từ phân phối thực nghiệm trong tập dữ liệu:

| Tên đặc trưng | Công thức | Ý nghĩa |
|---|---|---|
| `{COL}_freq` | Số lần xuất hiện của giá trị trong tập dữ liệu | Tần suất tuyệt đối |
| `{COL}_freq_log1p` | $\ln(1 + \text{\{COL\}\_freq})$ | Tần suất sau biến đổi log |

Đặc trưng tần suất phân biệt *shell provider* (nhà cung cấp ma xuất hiện rất ít lần, thường là dấu hiệu gian lận) với provider hợp lệ đang hoạt động bình thường. Quy trình này được áp dụng cho tối đa 20 cột định danh đầu tiên để kiểm soát số chiều.

### 3.2. Lựa chọn Đặc trưng Đưa vào Mô hình

Sau khi trích xuất, đặc trưng được phân thành hai nhóm:

- **Biến số** (*numeric*): Tất cả cột có kiểu dữ liệu số, bao gồm các đặc trưng nghiệp vụ dẫn xuất và đặc trưng tần suất log.
- **Biến phân loại** (*categorical*): Các cột mã y tế (khớp với `ICD`, `DGNS`, `PRCDR`, `DRG`, `HCPCS`, hậu tố `_CD` hoặc `CODE`) và cột tần suất thô (`_freq`).

Các cột định danh thô (`CLM_ID`, `PRVDR_NUM`, `AT_PHYSN_NPI`, v.v.) bị loại trực tiếp ra khỏi tập đặc trưng do tính duy nhất cao không mang giá trị thống kê — nhưng các đặc trưng tần suất được dẫn xuất từ chúng vẫn được giữ lại.

Kết quả: **77 biến số** và **17 biến phân loại**, tổng cộng **94 đặc trưng** đưa vào mô hình.

### 3.3. Pipeline Tiền xử lý

Pipeline tiền xử lý được xây dựng theo kiến trúc `ColumnTransformer` với hai nhánh xử lý độc lập và song song, được huấn luyện (`fit`) hoàn toàn trên tập train trước khi áp dụng (`transform`) lên tập test — đảm bảo không có hiện tượng *data leakage*.

**Bảng 3.2.** Cấu hình pipeline tiền xử lý.

| Nhánh | Bước 1: Xử lý khuyết | Bước 2: Mã hóa | Bước 3: Chuẩn hóa |
|---|---|---|---|
| **Biến số** | `SimpleImputer(strategy="median")` | — | `StandardScaler()` |
| **Biến phân loại** | `SimpleImputer(strategy="constant", fill_value="__MISSING__")` | `FrequencyEncoder()` | `StandardScaler()` |

**Lựa chọn chiến lược imputation**: Median được sử dụng thay cho mean đối với biến số vì phân phối chi phí và thời gian điều trị thường bị lệch phải mạnh và có các điểm ngoại lai — trong trường hợp đó mean bị kéo lệch theo outlier, trong khi median bền vững hơn.

**FrequencyEncoder**: Thay vì label encoding — vốn tạo thứ tự giả (artificial ordinal relationship) giữa các giá trị không có quan hệ thứ bậc — mỗi giá trị phân loại được ánh xạ thành tần suất tương đối của nó trong tập huấn luyện:

$$\text{freq\_encode}(v) = \frac{\text{count}(v \text{ trong tập train})}{N_{\text{train}}}$$

Phương pháp này phù hợp đặc biệt với bài toán phát hiện bất thường: giá trị hiếm gặp nhận giá trị số thấp, phản ánh trực giác rằng hành vi ít phổ biến tiềm ẩn rủi ro cao hơn. Giá trị không xuất hiện trong tập train (*unseen values*) được gán tần suất bằng 0.

**Chuẩn hóa z-score**: Sau imputation và encoding, tất cả đặc trưng được chuẩn hóa về phân phối chuẩn ($\mu = 0$, $\sigma = 1$) bằng `StandardScaler`. Bước này cần thiết để các mô hình dựa trên khoảng cách Euclidean (CBLOF, OCSVM) không bị thiên vị bởi sự khác biệt thang đo giữa các đặc trưng.

---

## Phần 4: Huấn luyện và Suy luận Bốn Mô hình

### 4.1. Tổng quan Thiết kế Thực nghiệm

Bốn thuật toán phát hiện bất thường không giám sát được lựa chọn nhằm phủ rộng bốn nguyên lý phát hiện bất thường khác nhau: cô lập ngẫu nhiên, phân cụm, ranh giới một lớp, và thống kê phân phối đuôi. Tất cả mô hình được huấn luyện trên cùng tập X_train đã tiền xử lý (2.100 × 94) và đánh giá trên cùng tập X_test (900 × 94). Tỉ lệ nhiễm `contamination = 0,05` được đặt thống nhất cho tất cả mô hình, phù hợp với tỉ lệ bất thường mô phỏng được sử dụng trong bước tạo nhãn.

### 4.2. Mô tả Chi tiết Bốn Thuật toán và Tham số

#### 4.2.1. Isolation Forest

**Nguyên lý**: Isolation Forest [Liu et al., 2008] phát hiện bất thường dựa trên tính chất *cô lập*: điểm bất thường nằm cô lập trong không gian đặc trưng nên dễ bị tách biệt hơn — nghĩa là đường đi từ gốc cây đến điểm đó ngắn hơn — so với điểm bình thường nằm trong vùng dày đặc. Mô hình xây dựng một rừng gồm $T$ cây *iTree*, mỗi cây được tạo bằng cách lặp lại quá trình chọn ngẫu nhiên một đặc trưng và một ngưỡng tách trong phạm vi giá trị của đặc trưng đó. Điểm bất thường của quan sát $x$ được tính là:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

trong đó $E[h(x)]$ là độ sâu trung bình của $x$ qua tất cả các cây và $c(n)$ là độ sâu trung bình kỳ vọng của một BST với $n$ điểm (hệ số chuẩn hóa).

**Cấu hình tham số** được sử dụng trong thực nghiệm:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `n_estimators` | 200 | Số cây trong rừng; giá trị 200 cho kết quả ổn định hơn mặc định 100 |
| `max_samples` | `"auto"` | Mỗi cây huấn luyện trên $\min(256, N_{\text{train}})$ mẫu con; không load toàn bộ tập |
| `max_features` | 1,0 | Tỉ lệ đặc trưng được xem xét khi tách; 1,0 nghĩa là dùng tất cả 94 đặc trưng |
| `contamination` | 0,05 | Tỉ lệ bất thường kỳ vọng; dùng để xác định ngưỡng phân loại nhị phân |
| `random_state` | 42 | Hạt nhân ngẫu nhiên đảm bảo tính tái lập |
| `n_jobs` | −1 | Sử dụng toàn bộ lõi CPU có sẵn (song song hóa) |

**Quy ước điểm**: `decision_function` của scikit-learn trả về giá trị dương cho điểm bình thường và âm cho điểm bất thường. Trong pipeline này, điểm được đảo dấu (`scores = -decision_function`) để thống nhất quy ước "điểm cao = bất thường hơn" với các mô hình khác.

**Không gian tham số trong grid search** (khi bật `--tune`): `n_estimators` ∈ {100, 200, 300}, `max_samples` ∈ {"auto", 0.5, 0.8}, `contamination` ∈ {0,01; 0,03; 0,05; 0,10} — tổng 36 tổ hợp.

#### 4.2.2. CBLOF — Cluster-Based Local Outlier Factor

**Nguyên lý**: CBLOF [He et al., 2003] phân cụm dữ liệu thành các cụm lớn (*large clusters* — chứa phần lớn dữ liệu bình thường) và cụm nhỏ (*small clusters* — có thể là nhóm bất thường). Điểm bất thường của quan sát $x$ thuộc cụm $C_k$ được tính dựa trên tích của kích thước cụm và khoảng cách đến cụm lớn gần nhất:

$$\text{CBLOF}(x) = |C_k| \times \text{dist}(x, C_{\text{nearest large}})$$

Nếu $x$ thuộc cụm lớn, khoảng cách được tính đến tâm cụm đó; nếu thuộc cụm nhỏ, khoảng cách được tính đến cụm lớn gần nhất. Cơ chế này đặc biệt hiệu quả khi gian lận tập trung thành các nhóm nhỏ cô lập xa trung tâm hành vi bình thường.

**Cấu hình tham số** được sử dụng trong thực nghiệm:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `n_clusters` | 8 | Số cụm k-Means; giá trị 8 là cân bằng giữa độ phân giải và ổn định |
| `alpha` | 0,9 | Ngưỡng tích lũy để phân biệt cụm lớn/nhỏ: cụm lớn là những cụm chiếm tổng cộng 90% dữ liệu khi sắp xếp theo kích thước giảm dần |
| `beta` | 5 | Hệ số: nếu kích thước cụm nhỏ nhất trong nhóm "lớn" ÷ kích thước cụm lớn nhất trong nhóm "nhỏ" ≥ beta, phân loại cũng được chấp nhận |
| `contamination` | 0,05 | Tỉ lệ bất thường kỳ vọng; xác định ngưỡng `threshold_` |
| `use_weights` | False | Không sử dụng trọng số khoảng cách (dùng khoảng cách thuần Euclidean) |
| `random_state` | 42 | Hạt nhân ngẫu nhiên cho k-Means |

**Quy ước điểm**: Thư viện PyOD sử dụng quy ước "điểm cao = bất thường hơn". `decision_function` trả về điểm CBLOF gốc; `threshold_` là giá trị ngưỡng học được từ tập train (giá trị phân vị $1 - \tau$ của phân phối điểm trên tập train).

**Không gian tham số trong grid search**: `n_clusters` ∈ {4, 8, 12, 16}, `alpha` ∈ {0,8; 0,9}, `beta` ∈ {3, 5}, `contamination` ∈ {0,01; 0,03; 0,05; 0,10} — tổng 64 tổ hợp.

#### 4.2.3. One-Class SVM (OCSVM)

**Nguyên lý**: One-Class SVM [Schölkopf et al., 2001] học một siêu mặt phân cách (*hyperplane*) trong không gian đặc trưng kernel nhằm bao trọn vùng chứa $(1-\nu)$ phần dữ liệu huấn luyện, tách biệt với gốc tọa độ. Bài toán tối ưu hóa được phát biểu là:

$$\min_{w, \xi, \rho} \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^{n} \xi_i$$
$$\text{s.t. } \langle w, \phi(x_i) \rangle \geq \rho - \xi_i, \quad \xi_i \geq 0$$

trong đó $\phi(\cdot)$ là ánh xạ đặc trưng kernel và $\rho$ là khoảng cách từ gốc tọa độ đến siêu mặt phân cách. Quan sát có $\langle w, \phi(x) \rangle < \rho$ được phân loại là bất thường.

**Cấu hình tham số** được sử dụng trong thực nghiệm:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `kernel` | `"rbf"` | Kernel RBF (Radial Basis Function) $K(x,y) = \exp(-\gamma \|x-y\|^2)$; cho phép học biên phi tuyến |
| `nu` | 0,05 | Cận trên của tỉ lệ điểm ngoại lệ (*outlier fraction*) và cận dưới của tỉ lệ support vector; đặt bằng `contamination` |
| `gamma` | `"scale"` | $\gamma = 1 / (n\_features \times \text{Var}(X))$; tự động điều chỉnh theo số chiều và phương sai dữ liệu |

*Lưu ý*: `random_state` không có tác dụng với `OneClassSVM` trong scikit-learn vì thuật toán là tất định (*deterministic*) với cùng tập dữ liệu.

**Quy ước điểm**: Tương tự Isolation Forest, `decision_function` trả về giá trị dương cho bình thường và âm cho bất thường. Điểm được đảo dấu trong pipeline.

**Không gian tham số trong grid search**: `nu` ∈ {0,01; 0,03; 0,05; 0,10}, `gamma` ∈ {"scale", "auto", 0,001, 0,01} — tổng 16 tổ hợp. Không gian tìm kiếm được giới hạn để kiểm soát thời gian chạy do OCSVM có độ phức tạp $\mathcal{O}(n^2)$ đến $\mathcal{O}(n^3)$ theo số mẫu.

#### 4.2.4. ECOD — Empirical Cumulative Distribution-based Outlier Detection

**Nguyên lý**: ECOD [Li et al., 2022] phát hiện bất thường dựa trên xác suất đuôi phân phối. Với mỗi chiều đặc trưng $j$, hàm phân phối tích lũy thực nghiệm (ECDF) được ước tính từ tập huấn luyện. Điểm bất thường của quan sát $x$ được tính là:

$$\text{ECOD}(x) = -\sum_{j=1}^{d} \log \hat{p}_j(x_j)$$

trong đó $\hat{p}_j(x_j) = \min\left[\hat{F}_j(x_j),\ 1 - \hat{F}_j(x_j)\right]$ là xác suất đuôi nhỏ nhất của giá trị $x_j$ theo chiều $j$, và $d$ là số chiều đặc trưng. Điểm cao thể hiện quan sát nằm ở vùng đuôi của nhiều đặc trưng đồng thời — dấu hiệu của bất thường đa biến.

**Cấu hình tham số** được sử dụng trong thực nghiệm:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `contamination` | 0,05 | Tỉ lệ bất thường kỳ vọng; xác định ngưỡng `threshold_` (phân vị $1 - \tau$ của điểm trên tập train) |

ECOD thực tế không có siêu tham số phức tạp cần điều chỉnh. Thuật toán gần như tất định (*nearly deterministic*) và không sử dụng `random_state`.

**Quy ước điểm**: Giống CBLOF (PyOD convention), điểm cao hơn nghĩa là bất thường hơn. `threshold_` là ngưỡng phân loại nhị phân.

**Không gian tham số trong grid search**: Chỉ có `contamination` ∈ {0,01; 0,03; 0,05; 0,10} — tổng 4 tổ hợp.

### 4.3. Cơ chế Tự động Lựa chọn Mô hình

Thay vì cố định một mô hình duy nhất, hệ thống triển khai quy trình lựa chọn mô hình hai tầng nhằm thích nghi tự động với đặc điểm từng tập dữ liệu.

#### 4.3.1. Tầng 1: Đánh giá Heuristic Trước Huấn luyện

Mỗi mô hình được chấm điểm phù hợp (*suitability score*) trên thang 0–100 dựa trên hồ sơ dữ liệu (*data profile*) gồm: số hồ sơ, số chiều đặc trưng, tỉ lệ đặc trưng phân loại, tỉ lệ khuyết, và tỉ lệ nhiễm kỳ vọng. Bảng 4.1 trình bày ma trận điểm heuristic.

**Bảng 4.1.** Ma trận điểm heuristic cho từng mô hình theo đặc điểm dữ liệu.

| Điều kiện | IF | CBLOF | OCSVM | ECOD |
|---|:---:|:---:|:---:|:---:|
| Điểm cơ bản | +18 | +12 | +5 | +14 |
| $n \geq 10.000$ | +8 | — | — | +6 |
| $1.000 \leq n \leq 100.000$ | — | +10 | — | — |
| $n > 30.000$ | — | — | −18 | — |
| $n \leq 10.000$ | — | — | +12 | — |
| $d \geq 30$ | +5 | — | — | — |
| $d \leq 60$ | — | +8 | — | — |
| $d > 60$ | — | −8 | — | — |
| $d \leq 40$ | — | — | +7 | — |
| $d > 40$ | — | — | −6 | — |
| Tỉ lệ numeric $\geq 0{,}60$ | — | — | — | +12 |
| Tỉ lệ categorical $> 0{,}30$ | +4 | — | — | — |
| Tỉ lệ categorical $> 0{,}50$ | — | −6 | — | −8 |
| $\tau > 0{,}10$ | — | — | −5 | — |
| $\tau \leq 0{,}05$ | — | — | — | +4 |

*Ghi chú: IF = Isolation Forest, CBLOF = Cluster-Based Local Outlier Factor, OCSVM = One-Class SVM, ECOD = Empirical Cumulative Distribution-based Outlier Detection. Điểm cơ sở là 50; điểm cuối được cắt trong khoảng [0, 100].*

Áp dụng vào tập thực nghiệm (n = 3.000, d = 94, tỉ lệ numeric = 81,9%, categorical = 18,1%, $\tau = 0,05$):

**Bảng 4.2.** Kết quả chấm điểm heuristic trên tập thực nghiệm.

| Hạng | Mô hình | Điểm phù hợp | Lý do chính |
|:---:|---|:---:|---|
| 1 | ECOD | 80,0 | Dữ liệu numeric-heavy (81,9%) phù hợp giả định phân phối đuôi; $\tau \leq 0{,}05$ |
| 2 | Isolation Forest | 73,0 | Baseline tổng quát; xử lý tốt không gian 94 chiều |
| 3 | CBLOF | 64,0 | Kích thước mẫu phù hợp; tuy nhiên $d = 94 > 60$ làm giảm chất lượng phân cụm |
| 4 | OCSVM | 61,0 | $n = 3.000 \leq 10.000$ được điểm cộng; song $d = 94 > 40$ gây bất lợi |

#### 4.3.2. Tầng 2: Lựa chọn theo Metric Sau Huấn luyện

Khi có nhãn đánh giá, sau khi chạy cả bốn mô hình, mô hình tốt nhất được chọn theo thứ tự ưu tiên: **F1-Score → ROC-AUC → Average Precision**. F1 được đặt làm chỉ số ưu tiên do bài toán phát hiện gian lận yêu cầu cân bằng giữa hai yêu cầu đối lập: không bỏ sót trường hợp gian lận thực sự (Recall cao) và không điều tra oan hồ sơ bình thường (Precision cao).

#### 4.3.3. Ba Chế độ Vận hành

| Chế độ | Hành vi |
|---|---|
| `--model-mode all` | Chạy cả 4 mô hình; sử dụng khi cần bảng so sánh đầy đủ |
| `--model-mode recommended` | Chạy 2 mô hình có suitability score cao nhất theo heuristic |
| `--model-mode auto` | Có nhãn: chạy cả 4 → chọn theo metric; Không nhãn: chạy 1 mô hình heuristic tốt nhất |

### 4.4. Pipeline Suy luận Production

Hệ thống triển khai REST API (FastAPI) phục vụ suy luận thời gian thực. Mô hình và preprocessor được nạp một lần vào bộ nhớ (*lazy loading*) và tái sử dụng cho các request tiếp theo để tối thiểu hóa độ trễ.

Mỗi mô hình sử dụng thang điểm native khác nhau, do đó hệ thống chuyển đổi điểm về phần trăm rủi ro (0–100%) dựa trên ngưỡng quyết định riêng của từng mô hình:

**Bảng 4.3.** Quy tắc chuyển đổi điểm bất thường sang mức rủi ro theo từng mô hình.

| Mô hình | Thang điểm native | Công thức risk% | Phân loại |
|---|---|---|---|
| Isolation Forest | `decision_function` ∈ [−0,10; +0,14], dương = bình thường | $\text{risk} = \text{clip}\!\left(\left(0{,}5 - \frac{s}{2 \cdot |\text{offset\_}|}\right) \times 100,\ 0,\ 100\right)$ | HIGH: $s < 0$; MEDIUM: $0 \leq s < \text{band}$; LOW: $s \geq \text{band}$ |
| OCSVM | `decision_function`, dương = bình thường | Cùng công thức IF với `scale = |offset_|` | HIGH: $s < 0$; MEDIUM: $0 \leq s < \text{band}$; LOW: $s \geq \text{band}$ |
| CBLOF (PyOD) | Cao = bất thường, `threshold_` ≈ 8,28 | $\text{risk} = \text{clip}\!\left(\frac{s}{\text{threshold\_}} \times 50,\ 0,\ 100\right)$ | HIGH: $s \geq 1{,}5 \times \text{thr}$; MEDIUM: $\text{thr} \leq s < 1{,}5 \times \text{thr}$; LOW: $s < \text{thr}$ |
| ECOD (PyOD) | Cao = bất thường, `threshold_` ≈ 67,12 | Cùng công thức CBLOF | HIGH: $s \geq 1{,}5 \times \text{thr}$; MEDIUM: $\text{thr} \leq s < 1{,}5 \times \text{thr}$; LOW: $s < \text{thr}$ |

Điểm rủi ro ở cấp hồ sơ được tổng hợp thành **điểm rủi ro cấp nhà cung cấp** theo trung bình có trọng số, trong đó hồ sơ có risk% > 50 được tăng trọng số 1,5 lần. Nhà cung cấp có điểm tổng hợp > 70% được tự động chuyển sang hàng đợi điều tra với SLA từ 4–24 giờ tùy mức độ nghiêm trọng.

---

## Phần 5: Đánh giá và Trực quan hóa

### 5.1. Bộ Chỉ số Đánh giá

Bộ chỉ số được thiết kế phù hợp với hai đặc thù của bài toán: dữ liệu mất cân bằng nghiêm trọng (5% bất thường, 95% bình thường) và yêu cầu ứng dụng thực tế (danh sách ưu tiên điều tra giới hạn).

| Chỉ số | Công thức | Mục đích |
|---|---|---|
| ROC-AUC | $\int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$ | Khả năng xếp hạng tổng thể, độc lập ngưỡng |
| Average Precision (AP) | $\sum_k (R_k - R_{k-1}) P_k$ | AUC trên đường cong PR; đặc biệt tin cậy khi class imbalance cao |
| F1-Score | $\frac{2 \cdot P \cdot R}{P + R}$ | Cân bằng Precision–Recall tại ngưỡng phân loại |
| Precision | $\text{TP}/(\text{TP}+\text{FP})$ | Tỉ lệ điều tra đúng |
| Recall | $\text{TP}/(\text{TP}+\text{FN})$ | Tỉ lệ bất thường thực sự được phát hiện |
| Precision@K% | $\frac{1}{K}\sum_{i=1}^{K} y_{(i)}$ | Độ chính xác trong top K% hồ sơ điểm cao nhất |
| Thời gian chạy (giây) | Đo bằng `time.time()` | Khả năng triển khai thực tế |

Chỉ số Precision@K được tính với K ∈ {0,5%; 1%; 2%; 5%} (tương ứng top 4–5, 9, 18, 45 hồ sơ trong tập 900). Đây là chỉ số quan trọng nhất trong ứng dụng thực tiễn vì điều tra viên thường chỉ có khả năng xem xét một số hữu hạn hồ sơ ưu tiên cao nhất trong một đợt làm việc.

### 5.2. Kết quả Đánh giá Toàn diện

**Bảng 5.1.** Kết quả so sánh bốn mô hình trên tập kiểm tra (n = 900, tỉ lệ bất thường ≈ 5%).

| Mô hình | ROC-AUC | AP | Precision | Recall | **F1** | P@0,5% | P@1% | P@2% | P@5% | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECOD** | **0,874** | **0,455** | 0,360 | **0,585** | **0,446** | **1,000** | **0,778** | **0,722** | 0,467 | **0,246** |
| Isolation Forest | 0,824 | 0,311 | **0,380** | 0,358 | 0,369 | 0,500 | 0,444 | 0,500 | 0,333 | 0,651 |
| CBLOF | 0,828 | 0,292 | 0,375 | 0,340 | 0,356 | 0,500 | 0,444 | 0,389 | **0,378** | 8,327 |
| OCSVM | 0,788 | 0,240 | 0,279 | 0,415 | 0,333 | 0,250 | 0,444 | 0,389 | 0,289 | 0,104 |

*Ghi chú: Giá trị in đậm là tốt nhất trong từng chỉ số. Nhãn đánh giá là nhãn bất thường mô phỏng. Ma trận nhầm lẫn: ECOD (TN=792, FP=55, FN=22, TP=31); IF (TN=816, FP=31, FN=34, TP=19); CBLOF (TN=817, FP=30, FN=35, TP=18); OCSVM (TN=790, FP=57, FN=31, TP=22).*

### 5.3. Phân tích So sánh Các Mô hình

**ECOD** đạt hiệu suất tốt nhất trên hầu hết chỉ số. ROC-AUC = 0,874 phản ánh khả năng xếp hạng vượt trội: mô hình có xác suất 87,4% xếp một hồ sơ bất thường cao hơn một hồ sơ bình thường được chọn ngẫu nhiên. Recall = 0,585 cho thấy ECOD phát hiện được 58,5% hồ sơ bất thường thực sự (TP = 31/53), cao hơn đáng kể so với các mô hình còn lại. Đặc biệt, Precision@0,5% = 1,000 — toàn bộ hồ sơ trong nhóm ưu tiên cao nhất (top ~4–5 hồ sơ) đều là bất thường thực sự, và Precision@1% = 0,778 cho thấy khoảng 7 trong 9 hồ sơ trong top 1% được điều tra đúng. Đây là kết quả có giá trị ứng dụng cao. Ngoài ra, thời gian chạy 0,246 giây là cạnh tranh — ECOD nhanh hơn Isolation Forest (0,651s) và CBLOF (8,327s), mặc dù OCSVM trên tập nhỏ này nhanh hơn (0,104s).

**Isolation Forest** thể hiện là baseline mạnh và cân bằng: Precision cao nhất (0,380), đồng nghĩa với ít báo động giả nhất trong số các mô hình. Điều này quan trọng trong bối cảnh triển khai thực tế khi tài nguyên điều tra có hạn và chi phí điều tra sai cũng đáng kể. FP = 31 (thấp thứ hai sau CBLOF với FP = 30).

**CBLOF** cho thấy ROC-AUC tương đương Isolation Forest (0,828 so với 0,824) nhưng F1 thấp hơn do cả Precision và Recall đều thấp hơn một chút. Nhược điểm rõ nhất là thời gian chạy 8,327 giây — cao gấp 12–34 lần các mô hình còn lại — phản ánh gánh nặng tính toán của bước phân cụm k-Means ($\mathcal{O}(n \cdot k \cdot d \cdot \text{iter})$) bên trong. Tuy nhiên, cần lưu ý rằng trên tập dữ liệu đầy đủ (không giới hạn mẫu), CBLOF đạt F1 = 0,458 và ROC-AUC = 0,906 — cho thấy mô hình này được lợi nhiều hơn từ lượng dữ liệu lớn hơn do chất lượng phân cụm cải thiện.

**OCSVM** cho kết quả yếu nhất với ROC-AUC thấp nhất (0,788), F1 thấp nhất (0,333) và Precision@0,5% thấp nhất (0,250). Xu hướng tạo nhiều báo động giả (FP = 57, cao nhất) phù hợp với dự đoán lý thuyết: OCSVM học ranh giới trong không gian kernel 94 chiều gặp khó khăn do *curse of dimensionality*, khiến biên ranh giới học được kém chính xác.

### 5.4. Lựa chọn Mô hình Cuối cùng

Dựa trên cả hai tầng đánh giá, **ECOD** được hệ thống lựa chọn làm mô hình chính — nhất quán hoàn toàn giữa gợi ý heuristic (hạng 1, điểm 80,0) và lựa chọn theo metric thực nghiệm (F1 cao nhất = 0,446, ROC-AUC cao nhất = 0,874, AP cao nhất = 0,455). Sự nhất quán này xác nhận tính hiệu quả của chiến lược heuristic profiling trong việc dự đoán mô hình phù hợp trước khi huấn luyện — đặc biệt có giá trị trong tình huống thực tế không có nhãn đánh giá.

### 5.5. Điều chỉnh Siêu tham số

Khi bật cờ `--tune`, hệ thống thực hiện grid search toàn diện trên không gian tham số của cả bốn mô hình. Bảng 5.2 tóm tắt không gian tìm kiếm.

**Bảng 5.2.** Không gian siêu tham số trong grid search.

| Mô hình | Siêu tham số | Các giá trị thử nghiệm | Tổng tổ hợp |
|---|---|---|:---:|
| Isolation Forest | `n_estimators` | {100, 200, 300} | |
| | `max_samples` | {"auto", 0,5, 0,8} | |
| | `contamination` | {0,01; 0,03; 0,05; 0,10} | **36** |
| CBLOF | `n_clusters` | {4, 8, 12, 16} | |
| | `alpha` | {0,8; 0,9} | |
| | `beta` | {3, 5} | |
| | `contamination` | {0,01; 0,03; 0,05; 0,10} | **64** |
| OCSVM | `nu` | {0,01; 0,03; 0,05; 0,10} | |
| | `gamma` | {"scale", "auto", 0,001, 0,01} | **16** |
| ECOD | `contamination` | {0,01; 0,03; 0,05; 0,10} | **4** |

Kết quả grid search được lưu vào `outputs/tuning_results.csv` (toàn bộ 120 tổ hợp) và `outputs/best_tuning_results.csv` (tổ hợp tốt nhất của từng mô hình theo F1).

### 5.6. Biểu đồ Trực quan hóa

Bốn loại biểu đồ được tạo tự động và lưu vào thư mục `outputs/`:

**Đường cong ROC** (`roc_curves.png`): So sánh đồng thời đường cong ROC của bốn mô hình trên cùng hệ trục, với AUC hiển thị trong legend. ECOD nằm cao nhất và xa nhất so với đường cơ sở ngẫu nhiên (AUC = 0,5).

**Đường cong Precision-Recall** (`pr_curves.png`): Quan trọng hơn ROC khi tập dữ liệu mất cân bằng. Đường cong PR của ECOD (AP = 0,455) vượt trội rõ rệt, đặc biệt ở vùng Recall thấp — nghĩa là khi danh sách điều tra ngắn, ECOD có tỉ lệ đúng cao hơn đáng kể.

**Ma trận nhầm lẫn** (`confusion_matrices.png`): Heatmap (seaborn) bốn mô hình song song, hiển thị TP, FP, TN, FN với nhãn "Normal" và "Anomaly".

**Phân phối điểm bất thường** (`score_distributions.png`): Histogram chồng lớp hai nhóm Normal (xanh) và Anomaly (đỏ) theo mật độ xác suất. Mô hình tốt cho thấy hai phân phối tách biệt và ít chồng lấp. Biểu đồ này giúp lý giải trực quan tại sao ECOD đạt ROC-AUC và AP cao hơn.

### 5.7. Danh sách Hồ sơ Ưu tiên Điều tra

Hệ thống xuất danh sách 200 hồ sơ có rủi ro cao nhất (`top_suspicious_claims.csv`) được xếp hạng theo **điểm tổ hợp** (*ensemble rank score*) — trung bình thứ hạng phần trăm (*percentile rank*) của từng mô hình cho mỗi hồ sơ:

$$\text{ensemble\_rank}(x) = \frac{1}{M} \sum_{m=1}^{M} \text{rank\_pct}_m(x)$$

trong đó $M$ là số mô hình và $\text{rank\_pct}_m(x)$ là vị trí phần trăm của hồ sơ $x$ trong phân phối điểm của mô hình $m$. Cách tổng hợp này trung hòa hoàn toàn sự khác biệt về thang đo giữa các mô hình và ưu tiên các hồ sơ được đánh giá nhất quán là bất thường bởi nhiều mô hình độc lập — nâng cao độ tin cậy của danh sách điều tra.

---

## Phần 6: Tổng kết

### 6.1. Tổng hợp Kết quả

Pipeline học máy được xây dựng hoàn thành đầy đủ các mục tiêu đề ra. Trên tập thực nghiệm 3.000 hồ sơ với nhãn bất thường mô phỏng, ECOD nổi bật với ROC-AUC = 0,874, F1 = 0,446 và Precision@0,5% = 1,000 — cho thấy toàn bộ hồ sơ trong nhóm ưu tiên điều tra cao nhất đều là bất thường theo định nghĩa nghiệp vụ. Cơ chế lựa chọn mô hình hai tầng chứng tỏ tính hiệu quả khi gợi ý heuristic nhất quán với lựa chọn theo metric, xác nhận rằng phân tích đặc điểm dữ liệu trước huấn luyện là một bước định hướng có giá trị.

Kết quả cũng cho thấy sự đánh đổi rõ ràng giữa các mô hình: ECOD tối ưu về Recall và độ chính xác đầu danh sách; Isolation Forest tối ưu về Precision (ít báo động giả hơn); CBLOF cải thiện khi có nhiều dữ liệu hơn; OCSVM phù hợp với không gian đặc trưng ít chiều hơn. Sự đánh đổi này cần được cân nhắc tùy theo ngữ cảnh triển khai cụ thể.

### 6.2. Giới hạn

1. **Nhãn mô phỏng**: Kết quả đánh giá phụ thuộc hoàn toàn vào chất lượng các quy tắc tạo nhãn. Hiệu suất thực tế trên gian lận được xác minh có thể khác biệt đáng kể và chỉ có thể đánh giá khi có nhãn thực tế từ đơn vị kiểm toán chuyên nghiệp.

2. **OCSVM không mở rộng được**: Độ phức tạp tính toán $\mathcal{O}(n^2)$ đến $\mathcal{O}(n^3)$ khiến OCSVM không thực tiễn khi tập dữ liệu vượt ngưỡng 30.000 hồ sơ.

3. **Khả năng giải thích hạn chế**: Bốn thuật toán đều thuộc nhóm hộp đen hoặc nửa hộp đen. Hệ thống cung cấp giải thích dựa trên quy tắc cứng (giá trị ngưỡng của một số đặc trưng cụ thể), chưa áp dụng phương pháp model-agnostic như SHAP hay LIME để lượng hóa đóng góp của từng đặc trưng.

4. **Không có cơ chế phát hiện data drift**: Hệ thống chưa tích hợp module giám sát sự thay đổi phân phối dữ liệu theo thời gian. Nếu hành vi gian lận thay đổi hoặc cấu trúc dữ liệu đầu vào thay đổi, mô hình sẽ dần suy giảm hiệu quả mà không có cảnh báo tự động.

### 6.3. Hướng Phát triển Đề xuất

- **Tích hợp nhãn thực tế**: Thu thập nhãn gian lận được xác minh từ đơn vị kiểm toán để chuyển từ đánh giá mô phỏng sang đánh giá thực nghiệm thực sự. Đây là bước ưu tiên cao nhất để xác định giá trị thực của hệ thống.
- **Bổ sung giải thích SHAP**: Tính SHAP values cho từng dự đoán để cung cấp lý do cụ thể cho điều tra viên — "hồ sơ này bị gắn cờ vì chi phí mỗi ngày cao bất thường và bác sĩ hiếm gặp".
- **Giám sát drift tự động**: Triển khai PSI (Population Stability Index) hoặc Kolmogorov-Smirnov test định kỳ trên phân phối đặc trưng đầu vào để kích hoạt retrain tự động khi phát hiện drift.
- **Mở rộng phạm vi dữ liệu**: Tích hợp hồ sơ ngoại trú (*outpatient*) và dược phẩm (*pharmacy*) để xây dựng hồ sơ rủi ro toàn diện hơn cho từng nhà cung cấp dịch vụ y tế.
