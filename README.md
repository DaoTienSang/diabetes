# Hệ thống dự đoán biến chứng tiểu đường theo thời gian thực

Hệ thống này mô phỏng một **AI Extend Module** tích hợp vào hệ thống bệnh viện/phòng khám giả lập. Module nhận hồ sơ bệnh nhân từ Mock HIS, dự đoán nguy cơ biến chứng tiểu đường và hiển thị cảnh báo cho bác sĩ.

Dự án tập trung vào quy trình triển khai AI/MLOps từ dữ liệu, huấn luyện mô hình, serving API, dashboard, logging, monitoring đến retraining. Đây là demo học thuật, không phải hệ thống chẩn đoán y khoa chính thức và không xây lại toàn bộ HIS.

Repository chính thức:

```text
https://github.com/DaoTienSang/diabetes.git
```

## 2. Mục tiêu của dự án

Dự án hướng đến việc xây dựng một module AI mở rộng cho hệ thống bệnh viện/phòng khám, có khả năng nhận hồ sơ bệnh nhân mới, dự đoán nguy cơ biến chứng tiểu đường type 2 theo nhiều nhãn và hiển thị cảnh báo cho bác sĩ trên Dashboard.

Các mục tiêu chính:

- Dự đoán nguy cơ biến chứng tiểu đường type 2.
- Trả risk score cho nhiều biến chứng cùng lúc.
- Hỗ trợ bác sĩ theo dõi bệnh nhân có nguy cơ cao.
- Mô phỏng real-time bằng Mock HIS.
- Có API, Dashboard, Database, Logging, Monitoring và Retraining.
- Thể hiện quy trình MLOps từ train model đến deploy và monitor.

## 3. Kiến trúc tổng quan

### 3.1. Data_And_Training

Khối Data_And_Training gồm:

- Historical Diabetes Complication Data.
- Data Cleaning.
- DuckDB analytical store.
- Feature Builder.
- Multi-label targets.
- Training script.
- MLflow Tracking.
- MLflow Model Registry.

Dữ liệu raw được đặt tại `data/raw/data.csv`, sau đó được clean thành `data/processed/clean_data.csv`. Bước cleaning chuẩn hóa tên cột, xử lý `dia_life`, ép kiểu numeric và loại bỏ duplicate theo các cột y tế + target, không dựa vào `sl_no` và `name`. Việc loại bỏ duplicate giúp giảm rò rỉ train/test.

DuckDB có bảng kỹ thuật `du_lieu_huan_luyen` phục vụ training. Bảng này thuộc analytical store, khác với các bảng nghiệp vụ dùng cho API/Dashboard. Feature Builder đọc dữ liệu từ bảng này để tạo `x`, `y` cho bài toán multi-label.

### 3.2. Serving_And_Deployment

Khối Serving_And_Deployment gồm:

- GitHub Actions.
- pytest.
- Docker Image.
- Docker Compose.
- FastAPI.
- Alert Engine.
- Django Dashboard.
- Request logging.
- Health check.
- Prometheus.
- Grafana.

FastAPI là service dự đoán. API nhận 15 input feature, trả risk score/risk label/risk level cho 5 biến chứng, sinh cảnh báo và lưu log vào DuckDB.

Django Dashboard là giao diện nội bộ cho bác sĩ/admin xem bệnh nhân, lịch sử dự đoán, risk score và cảnh báo. Bệnh nhân không có tài khoản và không trực tiếp xem Dashboard.

MLflow, Prometheus và Grafana dùng UI mặc định. Project không code lại UI của các công cụ này. Docker Compose chạy local nhiều service gồm FastAPI, Django, MLflow, Prometheus và Grafana.

### 3.3. Monitoring_And_Retraining

Khối Monitoring_And_Retraining gồm:

- Evidently AI.
- Drift / Recall trigger threshold.
- Prefect retrain job.
- Training pipeline.
- Register new model version.
- Clinical production rules recall-focused.
- Promote to Production hoặc No Production.

Evidently tạo drift report và `reports/monitoring_summary.json`. Prefect đọc `retrain_flag` trong summary này. Nếu `retrain_flag = yes`, hệ thống train candidate model.

Candidate model chỉ được promote nếu pass clinical production rules recall-focused. Nếu candidate fail rules thì giữ nguyên model production cũ, không overwrite model đang chạy.

## 4. Vì sao đây là AI Extend Module

Dự án không xây lại hệ thống bệnh viện/phòng khám. Mock HIS đại diện cho hệ thống bệnh viện/phòng khám đã có sẵn.

AI Module chỉ nhận dữ liệu, dự đoán, cảnh báo và lưu log. Thiết kế này tránh overlap với hệ thống đang có, phù hợp tinh thần môn học là mở rộng module thông minh thay vì xây lại toàn bộ HIS.

## 5. Dataset

Dataset chính:

```text
Micro and Macro vascular complications in Type-II diabetes — Mendeley dsjcb6pyd8
```

Dataset này là dữ liệu dạng bảng. Mỗi dòng là một hồ sơ bệnh nhân. Đây không phải dữ liệu chuỗi thời gian và không phải dữ liệu streaming thật. Dataset được dùng để train model dự đoán biến chứng.

Hạn chế của dataset:

- Dataset có nhiều dòng trùng, cần drop duplicate trước khi train.
- Sau khi drop duplicate, số dòng giảm đáng kể.
- Target `PER VAS` rất lệch lớp.
- Dataset chỉ phục vụ demo học thuật, chưa đủ điều kiện triển khai y tế thật.

## 6. Input features

Các input feature:

- `age`: tuổi bệnh nhân.
- `sex`: giới tính đã mã hóa.
- `bmi`: chỉ số khối cơ thể.
- `sp`: huyết áp tâm thu.
- `bp`: huyết áp tâm trương.
- `hba1c`: chỉ số đường huyết trung bình.
- `fps`: đường huyết lúc đói.
- `pps`: đường huyết sau ăn.
- `family_h`: tiền sử gia đình.
- `onset_age`: tuổi khởi phát tiểu đường.
- `dia_life`: số năm mắc bệnh.
- `smoking`: tình trạng hút thuốc.
- `phy_act`: hoạt động thể chất.
- `med_use`: sử dụng thuốc.
- `med_adh`: tuân thủ thuốc.

## 7. Targets

Các target:

- `nep`: biến chứng thận.
- `neu`: biến chứng thần kinh.
- `ret`: biến chứng võng mạc.
- `cv`: biến chứng tim mạch.
- `per_vas`: biến chứng mạch máu ngoại biên.

Đây là bài toán **multi-label classification** vì một bệnh nhân có thể có nhiều biến chứng cùng lúc.

## 8. Vì sao dùng Multi-label Classification

Một bệnh nhân có thể vừa có nguy cơ biến chứng thận, vừa có nguy cơ tim mạch hoặc võng mạc. Single-label classification chỉ chọn một lớp duy nhất nên không phù hợp.

Multi-label classification cho phép dự đoán 5 nhãn độc lập cùng lúc. Output của hệ thống gồm risk score, risk label và risk level cho từng biến chứng.

## 9. Vì sao dùng Logistic Regression + Random Forest/XGBoost

Logistic Regression được dùng làm baseline. Random Forest/XGBoost phù hợp với dữ liệu bảng tabular, chạy nhanh, dễ deploy API và phù hợp demo real-time.

Dataset Mendeley dsjcb6pyd8 không có chuỗi thời gian theo nhiều lần khám của cùng một bệnh nhân. Vì vậy tree-based models như Random Forest/XGBoost là lựa chọn hợp lý hơn. Khi cần ưu tiên hạn chế bỏ sót, hệ thống có thể dùng `class_weight`, threshold hoặc production rules thiên về Recall.

## 10. Vì sao không dùng LSTM

Không dùng LSTM vì LSTM phù hợp dữ liệu chuỗi thời gian. Dataset Mendeley là dữ liệu bảng tĩnh, mỗi bệnh nhân là một dòng. Nếu dùng LSTM cho dữ liệu này sẽ không đúng bản chất thuật toán.

Vì vậy nhóm chọn Logistic Regression baseline và Random Forest/XGBoost cho main model.

## 11. Real-time được fake như thế nào

Dataset không phải streaming thật. Nhóm mô phỏng real-time bằng Mock HIS.

Mock HIS đọc từng dòng hồ sơ bệnh nhân từ `clean_data.csv`, sau đó gửi từng hồ sơ đến FastAPI `/predict`. Có delay giữa các request để giống dữ liệu mới phát sinh. Mock HIS không gửi target sang API để tránh data leakage.

Chạy Mock HIS local:

```bash
python -m src.mock_his.stream --limit 5 --delay 1
```

Chạy Mock HIS trong Docker:

```bash
docker compose run --rm fastapi python -m src.mock_his.stream --url http://fastapi:8000/predict --limit 5 --delay 1
```

## 12. Cấu trúc project

```text
diabetes/
├── data/
│   ├── raw/
│   └── processed/
├── db/
├── models/
├── reports/
├── src/
│   ├── domain/
│   ├── data/
│   ├── db/
│   ├── features/
│   ├── train/
│   ├── predict/
│   ├── api/
│   ├── mock_his/
│   ├── monitoring/
│   └── retrain/
├── dashboard/
├── docker/
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Giải thích nhanh:

- `src/domain`: domain entities bám Class Diagram.
- `src/data`: load và clean data.
- `src/db`: DuckDB schema, repo, analytics store.
- `src/features`: Feature Builder.
- `src/train`: metrics, train models, MLflow utils.
- `src/predict`: Alert Engine.
- `src/api`: FastAPI.
- `src/mock_his`: fake real-time.
- `src/monitoring`: Evidently drift monitoring.
- `src/retrain`: Prefect retraining flow.
- `dashboard`: Django Dashboard cho bác sĩ.
- `docker`: Prometheus/Grafana config.
- `tests`: pytest.

## 13. Domain entities và Class Diagram

Project có `src/domain/entities.py` để bám Class Diagram. Các domain entities gồm `NguoiDung`, `BacSi`, `QuanTriVien`, `BenhNhan`, `HoSoLamSang`, `MoHinhDuDoan` và các entity hỗ trợ khác.

Quan hệ kế thừa:

- `BacSi`, `QuanTriVien` kế thừa `NguoiDung`.
- `LogisticRegressionModel`, `RandomForestModel`, `XGBoostModel` kế thừa `MoHinhDuDoan`.

Logic chính vẫn viết bằng hàm để code dễ hiểu và dễ test. Domain class chủ yếu giúp code bám thiết kế OOP, không làm phức tạp nghiệp vụ.

## 14. Database và DuckDB

### 14.1. Bảng analytical phục vụ training

Bảng `du_lieu_huan_luyen` dùng trong Data_And_Training. Bảng này load từ `clean_data.csv`. Feature Builder đọc bảng này để train model.

### 14.2. Bảng nghiệp vụ phục vụ API/Dashboard

Các bảng nghiệp vụ:

- `benh_nhan`
- `ho_so_lam_sang`
- `model_version`
- `ket_qua_du_doan`
- `chi_tiet_risk_score`
- `canh_bao`
- `request_log`
- `monitoring_log`

Database Diagram không có kế thừa. Database chỉ có bảng, cột, PK, FK và quan hệ. API lưu prediction và risk score vào DuckDB. Dashboard đọc DuckDB qua `src.db.repo`.

## 15. Cài môi trường

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 16. Đặt dataset

Đặt file dataset vào:

```text
data/raw/data.csv
```

Không nên commit dataset thật lên GitHub nếu không cần. Nếu chưa có dataset thật, CI/test sẽ dùng sample data nhỏ để kiểm tra pipeline.

## 17. Chạy Data Cleaning

```bash
python -m src.data.clean_data
```

Output:

- `data/processed/clean_data.csv`
- `reports/duplicate_rows.csv` nếu có duplicate

Bước này chuẩn hóa tên cột, xử lý `dia_life`, drop duplicate theo các cột y tế + target, bỏ qua `sl_no/name`, và không dùng `sl_no/name` làm feature.

## 18. Load clean data vào DuckDB

Nếu module có main:

```bash
python -m src.db.analytics
```

Lệnh Python dùng trực tiếp:

```bash
python - <<'PY'
from src.db.analytics import load_clean_to_duckdb
load_clean_to_duckdb()
print("duckdb training data ready")
PY
```

Output là bảng `du_lieu_huan_luyen` trong `db/app.duckdb`.

## 19. Chạy EDA

Lệnh EDA:

```bash
python -m src.data.eda
```

Output dự kiến:

- `reports/target_counts.png`
- `reports/hist_age.png`
- `reports/hist_bmi.png`
- `reports/hist_hba1c.png`
- `reports/hist_fps.png`
- `reports/hist_pps.png`
- `reports/hist_dia_life.png`
- `reports/corr.png`

## 20. Train model

```bash
python -m src.train.train_models
```

Output:

- `models/best_model.pkl`
- `models/meta.json`
- `reports/metrics.csv`
- `mlruns/` nếu MLflow local được dùng

Metrics:

- `exact_acc`: đúng toàn bộ 5 nhãn mới tính đúng.
- `label_acc`: trung bình từng nhãn đúng/sai.
- `f1_micro`: đánh giá tổng thể.
- `f1_macro`: trung bình từng biến chứng.
- `recall_macro`: quan trọng trong y tế để hạn chế bỏ sót.
- `hamming_loss`: tỷ lệ nhãn sai.

Không nên chỉ nhìn `exact_acc` vì multi-label exact accuracy thường thấp và rất khắt khe.

## 21. MLflow

MLflow dùng để:

- log params
- log metrics
- log artifacts
- log model
- quản lý model version nếu registry dùng được

Mở MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

URL:

```text
http://localhost:5000
```

Project không code lại MLflow UI, chỉ dùng UI mặc định. Nếu MLflow server không bật, train vẫn dùng local `mlruns` hoặc skip an toàn.

## 22. Chạy FastAPI

```bash
uvicorn src.api.main:app --reload
```

URL:

```text
http://localhost:8000
```

Endpoints:

- `GET /health`
- `POST /predict`
- `GET /metrics`

Health check:

```bash
curl http://localhost:8000/health
```

Ví dụ gọi `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "BN_DEMO_001",
    "name": "Demo Patient",
    "age": 45,
    "sex": 1,
    "bmi": 26.5,
    "sp": 130,
    "bp": 80,
    "hba1c": 8.1,
    "fps": 145,
    "pps": 220,
    "family_h": 1,
    "onset_age": 38,
    "dia_life": 7,
    "smoking": 0,
    "phy_act": 1,
    "med_use": 1,
    "med_adh": 2
  }'
```

Response gồm:

- `patient_id`
- `prediction_id`
- `risk_scores`
- `risk_labels`
- `risk_levels`
- `warning_level`
- `warning_message`
- `model_version`
- `timestamp`

## 23. Alert Engine

Alert Engine chuyển risk score thành risk level và warning message:

- Risk score `< 0.4`: `low`
- Risk score `0.4` đến `< 0.7`: `medium`
- Risk score `>= 0.7`: `high`

Nếu có biến chứng `high` thì tạo cảnh báo high. Nếu không có high nhưng có `medium` thì tạo cảnh báo medium. Nếu toàn low thì message là nguy cơ thấp.

Mapping biến chứng:

- `nep`: biến chứng thận
- `neu`: biến chứng thần kinh
- `ret`: biến chứng võng mạc
- `cv`: biến chứng tim mạch
- `per_vas`: biến chứng mạch máu ngoại biên

## 24. Django Dashboard

Dashboard chỉ dành cho bác sĩ/admin.

Không làm:

- không có đăng ký công khai
- không có tài khoản bệnh nhân
- không làm Patient Portal
- bệnh nhân không trực tiếp xem risk score

Tạo user demo:

```bash
python dashboard/manage.py migrate
python dashboard/manage.py create_demo_users
```

Chạy Dashboard:

```bash
python dashboard/manage.py runserver 8001
```

URL:

```text
http://localhost:8001
```

Tài khoản demo:

- `doctor / doctor123`
- `admin / admin123`

Các trang:

- `/login/`
- `/`
- `/patients/`
- `/patients/<ma_benh_nhan>/`
- `/alerts/`
- `/predictions/<ma_ket_qua>/`

Dashboard đọc dữ liệu bệnh nhân/dự đoán/cảnh báo từ DuckDB qua `src.db.repo`. Django SQLite chỉ dùng cho auth/session nếu cần.

## 25. Mock HIS fake real-time

Khi FastAPI đang chạy local:

```bash
python -m src.mock_his.stream --limit 5 --delay 1
```

Chỉ định URL:

```bash
python -m src.mock_his.stream --url http://localhost:8000/predict --limit 5 --delay 1
```

Mock HIS không gửi target. Module này chỉ gửi 15 input feature và mô phỏng real-time/streaming ở tầng API.

## 26. Prometheus

Prometheus scrape FastAPI `/metrics`.

Nếu chạy local không Docker, FastAPI metrics ở:

```text
http://localhost:8000/metrics
```

Nếu chạy Docker, Prometheus UI ở:

```text
http://localhost:9090
```

Project không code lại Prometheus UI.

## 27. Grafana

Grafana dùng để xem dashboard monitoring từ Prometheus.

Nếu chạy Docker:

- URL: `http://localhost:3000`
- Tài khoản: `admin / admin`

Project không code lại Grafana UI. Grafana dùng datasource Prometheus.

## 28. Evidently drift monitoring

```bash
python -m src.monitoring.drift
```

Output:

- `reports/drift_report.html` hoặc `reports/drift_report_skip.txt`
- `reports/monitoring_summary.json`
- `data/processed/current_data.csv` nếu chưa có

Evidently so sánh reference data và current data. Reference là `clean_data.csv`. Current ưu tiên lấy từ logs/hồ sơ mới trong DuckDB, nếu không có thì tạo demo `current_data.csv`.

`monitoring_summary.json` có `retrain_flag`. Nếu drift cao hoặc recall thấp thì `retrain_flag = yes`. Nếu Evidently lỗi version thì tạo skip file nhưng pipeline không crash.

## 29. Prefect retraining

Chạy retrain flow:

```bash
python -m src.retrain.flow
```

Force retrain:

```bash
python -m src.retrain.flow --force
```

Output:

- `models/candidate_model.pkl`
- `reports/candidate_metrics.csv`
- `reports/retrain_summary.json`
- `models/best_model.pkl` nếu candidate pass rules

Prefect đọc `reports/monitoring_summary.json`. Nếu `retrain_flag = no` thì không retrain. Nếu `retrain_flag = yes` thì train candidate model.

Candidate model phải pass clinical production rules recall-focused. Nếu pass thì promote. Nếu fail thì giữ production model cũ, không overwrite model tốt đang chạy.

## 30. Docker Compose

Chuẩn bị trước Docker nếu chạy local:

```bash
python -m src.data.clean_data
python -m src.train.train_models
python dashboard/manage.py migrate
python dashboard/manage.py create_demo_users
```

Chạy Docker Compose:

```bash
docker compose up --build
```

URL:

- FastAPI: `http://localhost:8000`
- FastAPI health: `http://localhost:8000/health`
- FastAPI metrics: `http://localhost:8000/metrics`
- Django Dashboard: `http://localhost:8001`
- MLflow UI: `http://localhost:5000`
- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000`

Dừng:

```bash
docker compose down
```

Mock HIS với Docker:

```bash
docker compose run --rm fastapi python -m src.mock_his.stream --url http://fastapi:8000/predict --limit 5 --delay 1
```

Docker Compose chạy 5 service:

- `fastapi`
- `django`
- `mlflow`
- `prometheus`
- `grafana`

## 31. GitHub Actions CI/CD

Repo chính thức:

```text
https://github.com/DaoTienSang/diabetes.git
```

Workflow:

```text
.github/workflows/ci.yml
```

Workflow chạy khi:

- push vào `main`
- pull_request vào `main`

Workflow kiểm tra:

- install dependencies
- tạo sample clean data cho CI
- chạy pytest
- import FastAPI
- import Django
- kiểm tra module train/db
- docker compose config
- docker build

Workflow không làm:

- không deploy production
- không push Docker image
- không cần secret
- không dùng dataset thật

Push lần đầu nếu cần:

```bash
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin https://github.com/DaoTienSang/diabetes.git
git push -u origin main
```

Nếu repo đã có remote:

```bash
git remote set-url origin https://github.com/DaoTienSang/diabetes.git
```

Push thay đổi:

```bash
git add .
git commit -m "Update project"
git push
```

Xem CI:

- Vào repo GitHub.
- Mở tab Actions.
- Chọn workflow `ci`.
- Xem job `test` và `docker`.

## 32. Chạy test

Chạy toàn bộ test:

```bash
pytest
```

Chạy từng nhóm:

```bash
pytest tests/test_data_cleaning.py
pytest tests/test_features.py
pytest tests/test_metrics.py
pytest tests/test_training.py
pytest tests/test_db.py
pytest tests/test_api.py
pytest tests/test_alert.py
pytest tests/test_mock_his.py
pytest tests/test_dashboard.py
pytest tests/test_dashboard_auth.py
pytest tests/test_drift.py
pytest tests/test_retrain_flow.py
pytest tests/test_docker_compose.py
pytest tests/test_ci_files.py
pytest tests/test_integration.py
```

Test đảm bảo data, model, API, DB, dashboard, monitoring, retraining, Docker và CI không bị vỡ. Integration test không chạy server thật, không dùng dataset thật và không chạy Docker Compose up.

## 33. Demo flow đề xuất khi bảo vệ

Thứ tự demo:

1. Cài môi trường.
2. Đặt dataset vào `data/raw/data.csv`.
3. Chạy data cleaning.
4. Load clean data vào DuckDB.
5. Train model.
6. Chạy FastAPI.
7. Gọi `/health`.
8. Gọi `/predict`.
9. Chạy Mock HIS gửi 5 hồ sơ.
10. Mở Django Dashboard.
11. Xem danh sách bệnh nhân.
12. Xem chi tiết bệnh nhân và risk score.
13. Xem cảnh báo.
14. Mở MLflow UI xem metrics/model.
15. Mở Prometheus xem metrics.
16. Mở Grafana xem dashboard.
17. Chạy Evidently drift report.
18. Chạy Prefect retrain flow.
19. Chạy pytest.
20. Chạy Docker Compose nếu cần.

## 34. Hạn chế

- Dataset public có số lượng không lớn.
- Dataset có duplicate nên phải drop duplicate.
- `PER VAS` bị lệch lớp.
- Dataset không phải real-time thật.
- Real-time chỉ được mô phỏng bằng Mock HIS.
- Model chỉ phục vụ demo học thuật.
- Kết quả không phải chẩn đoán y khoa.
- Cần dữ liệu bệnh viện thực tế và thẩm định chuyên môn nếu muốn triển khai thật.
- Bệnh nhân không trực tiếp xem kết quả để tránh hiểu nhầm risk score.

## 35. Troubleshooting

### Không tìm thấy data/raw/data.csv

Cách xử lý:

- Đặt dataset đúng path.
- Chạy lại `python -m src.data.clean_data`.

### Chưa có models/best_model.pkl

Cách xử lý:

- Chạy `python -m src.train.train_models`.

### /predict báo model missing

Cách xử lý:

- Kiểm tra `models/best_model.pkl`.
- Kiểm tra `models/meta.json`.

### Dashboard không có dữ liệu

Cách xử lý:

- Chạy FastAPI `/predict` hoặc Mock HIS để tạo prediction.
- Kiểm tra `db/app.duckdb`.

### Chưa đăng nhập được Dashboard

Cách xử lý:

- Chạy `python dashboard/manage.py migrate`.
- Chạy `python dashboard/manage.py create_demo_users`.

### Evidently lỗi version

Cách xử lý:

- Xem `reports/drift_report_skip.txt`.
- `monitoring_summary.json` vẫn được tạo nếu fallback hoạt động.

### Docker port bị trùng

Cách xử lý:

- Tắt service đang dùng port `8000`, `8001`, `5000`, `9090`, `3000`.
- Hoặc sửa ports trong `docker-compose.yml`.

### GitHub Actions fail vì thiếu dataset

Cách xử lý:

- CI không dùng dataset thật.
- Kiểm tra step tạo sample clean data trong workflow.

## 36. Các lệnh nhanh

```bash
pip install -r requirements.txt

python -m src.data.clean_data
python -m src.data.eda
python -m src.train.train_models

uvicorn src.api.main:app --reload

python -m src.mock_his.stream --limit 5 --delay 1

python dashboard/manage.py migrate
python dashboard/manage.py create_demo_users
python dashboard/manage.py runserver 8001

mlflow ui --host 0.0.0.0 --port 5000

python -m src.monitoring.drift
python -m src.retrain.flow

docker compose up --build

pytest
```
