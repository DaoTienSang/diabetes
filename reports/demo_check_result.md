# Demo Check Result

## Tổng quan
- Ngày chạy: 2026-05-14T20:25:42
- Python version: 3.10.11
- OS: Windows-10-10.0.26200-SP0
- Repo path: D:\lab_CNM\diabetes
- Dataset: real dataset at data/raw/data.csv
- Model hiện tại: random_forest
- DB path: db/app.duckdb

## Bảng trạng thái

| Module | Status | Ghi chú |
|---|---|---|
| Project structure | PASS | required files/folders exist |
| Imports | PASS | main modules import ok |
| Data cleaning | PASS | raw_shape (3068, 22) dup_no_id 2000 clean_shape (1068, 22) target_counts {'nep': 496, 'neu': 532, 'ret': 537, 'cv': 501, 'per_vas': 16} clean_shape=(1068, 22) |
| DuckDB analytics | PASS | training_shape=(1068, 21), counts={'rows': 1068, 'target_counts': {'nep': 496, 'neu': 532, 'ret': 537, 'cv': 501, 'per_vas': 16}} |
| Feature builder | PASS | x_shape=(1068, 15), y_shape=(1068, 5), train=(854, 15)/(854, 5), test=(214, 15)/(214, 5) |
| Training | PASS | model_name logistic_regression f1_micro 0.46021505376344085 f1_macro 0.40621877303674425 recall_macro 0.5070200342503164 model_name random_forest f1_micro 0.4740740740740741 f1_macro 0.3791014138840226 recall_macro 0.3675300614589828 model_name xgboost f1_micro 0.47342995169082125 f1_macro 0.3763927914476627 recall_macro 0.37472798896191967 best_model random_forest model_version 20260514202341 |
| Model output 5 targets | PASS | best_model=random_forest, model_version=20260514202341, pred_shape=(2, 5) |
| Database schema | PASS | tables=['benh_nhan', 'canh_bao', 'chi_tiet_risk_score', 'du_lieu_huan_luyen', 'ho_so_lam_sang', 'ket_qua_du_doan', 'model_version', 'monitoring_log', 'request_log'] |
| Database CRUD | PASS | counts={'benh_nhan': 8, 'ho_so_lam_sang': 7, 'model_version': 10, 'ket_qua_du_doan': 7, 'chi_tiet_risk_score': 35, 'canh_bao': 7, 'request_log': 6, 'monitoring_log': 3, 'du_lieu_huan_luyen': 1068}, prediction=kq_85503a4b8942 |
| Alert Engine | PASS | levels={'nep': 'high', 'neu': 'low', 'ret': 'medium', 'cv': 'low', 'per_vas': 'low'}, alert={'level': 'high', 'message': 'Bệnh nhân có nguy cơ cao về biến chứng thận. Cần ưu tiên theo dõi và kiểm tra lâm sàng.', 'high_targets': ['nep'], 'medium_targets': ['ret'], 'risk_levels': {'nep': 'high', 'neu': 'low', 'ret': 'medium', 'cv': 'low', 'per_vas': 'low'}} |
| FastAPI /health | PASS | {'status': 'ok'} |
| FastAPI /predict | PASS | {"patient_id": "BN_DEMO_API_202415", "prediction_id": "kq_45ece35a960d", "risk_scores": {"nep": 0.2438931871272052, "neu": 0.6138302410591249, "ret": 0.534658751779579, "cv": 0.25865793970103906, "per_vas": 0.0031746031746031746}, "risk_labels": {"nep": 0, "neu": 1, "ret": 1, "cv": 0, "per_vas": 0}, "risk_levels": {"nep": "low", "neu": "medium", "ret": "medium", "cv": "low", "per_vas": "low"}, "warning_level": "medium", "warning_message": "Bệnh nhân có nguy cơ trung bình về biến chứng thần kinh, |
| FastAPI /metrics | PASS | # HELP python_gc_objects_collected_total Objects collected during gc # TYPE python_gc_objects_collected_total counter python_gc_objects_collected_total{generation="0"} 756.0 python_gc_objects_collected_total{generation="1"} 64.0 python_gc_objects_collected_total{generation="2"} 30.0 # HELP python_gc |
| API DB logging | PASS | prediction_id=kq_45ece35a960d |
| Mock HIS | PASS | payload_keys=['patient_id', 'name', 'age', 'sex', 'bmi', 'sp', 'bp', 'hba1c', 'fps', 'pps', 'family_h', 'onset_age', 'dia_life', 'smoking', 'phy_act', 'med_use', 'med_adh'], target_leak=[] |
| Django login | PASS | migrate=True, users=True, before=302, login_page=200, login=True |
| Django Dashboard | PASS | home=200, patients=200, alerts=200 |
| No Patient Portal | PASS | register=404 |
| MLflow | PASS | tracking_uri=file:///D:/lab_CNM/diabetes/mlruns, registered_version=6 |
| Evidently | SKIP | Evidently report skipped but monitoring summary exists. report=False, skip=True, retrain_flag=yes, drift_share=1.0 |
| Prefect Retrain | PASS | monitoring_flag=yes, rules_pass=True, retrain_summary=True |
| Docker Compose config | PASS | services=True, ports=True, scrape=True, compose_config=True, err= |
| GitHub Actions config | PASS | workflow ci has pytest/docker/import checks |
| Pytest all | PASS | ======================== warnings summary =============================== C:\Users\Admin\AppData\Local\Programs\Python\Python310\lib\site-packages\requests\__init__.py:109   C:\Users\Admin\AppData\Local\Programs\Python\Python310\lib\site-packages\requests\__init__.py:109: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (7.4.3)/charset_normalizer (3.4.1) doesn't match a supported version!     warnings.warn(  tests/test_training.py: 40 warnings tests/test_training_mlflow.py: 30 warnings   C: |

## API demo response

```json
{
  "patient_id": "BN_DEMO_API_202415",
  "prediction_id": "kq_45ece35a960d",
  "risk_scores": {
    "nep": 0.2438931871272052,
    "neu": 0.6138302410591249,
    "ret": 0.534658751779579,
    "cv": 0.25865793970103906,
    "per_vas": 0.0031746031746031746
  },
  "risk_labels": {
    "nep": 0,
    "neu": 1,
    "ret": 1,
    "cv": 0,
    "per_vas": 0
  },
  "risk_levels": {
    "nep": "low",
    "neu": "medium",
    "ret": "medium",
    "cv": "low",
    "per_vas": "low"
  },
  "warning_level": "medium",
  "warning_message": "Bệnh nhân có nguy cơ trung bình về biến chứng thần kinh, biến chứng võng mạc. Cần tiếp tục theo dõi các chỉ số lâm sàng.",
  "model_version": "20260514202341",
  "timestamp": "2026-05-14T20:24:17.120661"
}
```

## Dashboard demo
- URL: http://localhost:8001
- Account doctor: doctor / doctor123
- Account admin: admin / admin123
- Các trang đã test: /login/, /, /patients/, /alerts/, /register/

## Monitoring demo
- /metrics status: PASS
- monitoring_summary.json: True
- retrain_flag: yes

## Lỗi còn lại
- Không có lỗi core còn lại.

## Kết luận
- demo_ready: yes
- core_failed: none
- optional_failed: none