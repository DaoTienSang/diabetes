# Demo Runbook

## 1. Giới thiệu kiến trúc AI Extend Module
- Lệnh chạy: `Không cần lệnh`
- Kết quả mong muốn: Nói rõ không xây lại HIS, chỉ mở rộng AI module.
- Nếu lỗi thì kiểm tra: Kiểm tra README mục kiến trúc.

## 2. Nói dataset Mendeley dsjcb6pyd8
- Lệnh chạy: `Không cần lệnh`
- Kết quả mong muốn: Giải thích dataset tabular, không phải streaming thật.
- Nếu lỗi thì kiểm tra: Kiểm tra data/raw/data.csv.

## 3. Chạy data cleaning
- Lệnh chạy: `python -m src.data.clean_data`
- Kết quả mong muốn: Tạo data/processed/clean_data.csv.
- Nếu lỗi thì kiểm tra: Kiểm tra raw path và cột dataset.

## 4. Load DuckDB
- Lệnh chạy:

```bash
python - <<'PY'
from src.db.analytics import load_clean_to_duckdb
load_clean_to_duckdb()
print('duckdb training data ready')
PY
```
- Kết quả mong muốn: Bảng du_lieu_huan_luyen có dữ liệu.
- Nếu lỗi thì kiểm tra: Kiểm tra db/app.duckdb.

## 5. Train model
- Lệnh chạy: `python -m src.train.train_models`
- Kết quả mong muốn: Tạo best_model.pkl, meta.json, metrics.csv.
- Nếu lỗi thì kiểm tra: Kiểm tra clean_data.csv và requirements.

## 6. Mở metrics.csv
- Lệnh chạy:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv('reports/metrics.csv'))
PY
```
- Kết quả mong muốn: Thấy logistic_regression/random_forest và metrics.
- Nếu lỗi thì kiểm tra: Kiểm tra reports/metrics.csv.

## 7. Chạy FastAPI
- Lệnh chạy: `uvicorn src.api.main:app --reload`
- Kết quả mong muốn: API chạy ở localhost:8000.
- Nếu lỗi thì kiểm tra: Kiểm tra models/best_model.pkl.

## 8. Gọi /health
- Lệnh chạy: `curl http://localhost:8000/health`
- Kết quả mong muốn: Trả {'status':'ok'}.
- Nếu lỗi thì kiểm tra: Kiểm tra FastAPI process.

## 9. Gọi /predict
- Lệnh chạy: `Dùng curl mẫu trong README mục FastAPI.`
- Kết quả mong muốn: Trả risk_scores/risk_levels/warning_message.
- Nếu lỗi thì kiểm tra: Kiểm tra model/meta.

## 10. Chạy Mock HIS gửi 5 hồ sơ
- Lệnh chạy: `python -m src.mock_his.stream --limit 5 --delay 1`
- Kết quả mong muốn: In send/status/level.
- Nếu lỗi thì kiểm tra: Kiểm tra FastAPI đang chạy.

## 11. Mở Django Dashboard
- Lệnh chạy: `python dashboard/manage.py runserver 8001`
- Kết quả mong muốn: Dashboard ở localhost:8001.
- Nếu lỗi thì kiểm tra: Chạy migrate và create_demo_users.

## 12. Login doctor / doctor123
- Lệnh chạy: `Mở http://localhost:8001/login/`
- Kết quả mong muốn: Đăng nhập thành công.
- Nếu lỗi thì kiểm tra: Chạy create_demo_users.

## 13. Xem danh sách bệnh nhân
- Lệnh chạy: `Mở /patients/`
- Kết quả mong muốn: Thấy bệnh nhân từ DuckDB.
- Nếu lỗi thì kiểm tra: Chạy /predict hoặc Mock HIS trước.

## 14. Xem chi tiết bệnh nhân
- Lệnh chạy: `Mở /patients/<ma_benh_nhan>/`
- Kết quả mong muốn: Thấy hồ sơ và prediction mới nhất.
- Nếu lỗi thì kiểm tra: Kiểm tra DB logs.

## 15. Xem risk score 5 biến chứng
- Lệnh chạy: `Trong trang chi tiết bệnh nhân`
- Kết quả mong muốn: Thấy nep, neu, ret, cv, per_vas.
- Nếu lỗi thì kiểm tra: Kiểm tra chi_tiet_risk_score.

## 16. Xem cảnh báo
- Lệnh chạy: `Mở /alerts/`
- Kết quả mong muốn: Thấy cảnh báo high/medium nếu có.
- Nếu lỗi thì kiểm tra: Gửi payload có risk cao.

## 17. Mở MLflow UI
- Lệnh chạy: `mlflow ui --host 0.0.0.0 --port 5000`
- Kết quả mong muốn: UI mặc định ở localhost:5000.
- Nếu lỗi thì kiểm tra: Kiểm tra mlruns.

## 18. Mở /metrics
- Lệnh chạy: `curl http://localhost:8000/metrics`
- Kết quả mong muốn: Thấy Prometheus metrics.
- Nếu lỗi thì kiểm tra: Kiểm tra FastAPI.

## 19. Mở Prometheus
- Lệnh chạy: `http://localhost:9090`
- Kết quả mong muốn: UI mặc định Prometheus.
- Nếu lỗi thì kiểm tra: Dùng Docker Compose.

## 20. Mở Grafana
- Lệnh chạy: `http://localhost:3000`
- Kết quả mong muốn: Login admin/admin, datasource Prometheus.
- Nếu lỗi thì kiểm tra: Dùng Docker Compose.

## 21. Chạy Evidently drift
- Lệnh chạy: `python -m src.monitoring.drift`
- Kết quả mong muốn: Tạo monitoring_summary.json.
- Nếu lỗi thì kiểm tra: Nếu lỗi version xem drift_report_skip.txt.

## 22. Xem monitoring_summary.json
- Lệnh chạy:

```bash
python - <<'PY'
import json
print(json.load(open('reports/monitoring_summary.json', encoding='utf-8')))
PY
```
- Kết quả mong muốn: Thấy retrain_flag/drift_share.
- Nếu lỗi thì kiểm tra: Chạy monitoring trước.

## 23. Chạy Prefect retrain
- Lệnh chạy: `python -m src.retrain.flow`
- Kết quả mong muốn: Tạo retrain_summary.json hoặc no_retrain.
- Nếu lỗi thì kiểm tra: Kiểm tra monitoring_summary.json.

## 24. Xem retrain_summary.json
- Lệnh chạy:

```bash
python - <<'PY'
import json
print(json.load(open('reports/retrain_summary.json', encoding='utf-8')))
PY
```
- Kết quả mong muốn: Thấy final_status/promoted/rule_result.
- Nếu lỗi thì kiểm tra: Chạy retrain flow trước.

## 25. Chạy docker compose config
- Lệnh chạy: `docker compose config`
- Kết quả mong muốn: Compose hợp lệ.
- Nếu lỗi thì kiểm tra: Kiểm tra Docker Desktop.

## 26. Nói GitHub Actions CI
- Lệnh chạy: `Mở .github/workflows/ci.yml`
- Kết quả mong muốn: Có pytest, docker compose config, docker build.
- Nếu lỗi thì kiểm tra: Kiểm tra tab Actions trên GitHub.

## 27. Chạy pytest
- Lệnh chạy: `pytest -q`
- Kết quả mong muốn: Toàn bộ test pass.
- Nếu lỗi thì kiểm tra: Nếu fail đọc tên test và sửa tối thiểu.

## 28. Kết luận hạn chế
- Lệnh chạy: `Không cần lệnh`
- Kết quả mong muốn: Nhấn mạnh demo học thuật, không phải chẩn đoán y khoa.
- Nếu lỗi thì kiểm tra: Mở README mục Hạn chế.
