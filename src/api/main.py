import time

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from starlette.responses import Response

from src.api.schema import PredictInput
from src.api.schema import PredictOutput
from src.api.service import predict_one


app = FastAPI(title="Diabetes Complication AI Extend Module")

request_count = Counter("request_count", "Total API requests")
predict_count = Counter("predict_count", "Total predict requests")
predict_error_count = Counter("predict_error_count", "Total predict errors")
predict_latency_seconds = Histogram("predict_latency_seconds", "Predict latency seconds")

app.state.db_path = "db/app.duckdb"
app.state.model_path = "models/best_model.pkl"
app.state.meta_path = "models/meta.json"


@app.get("/health")
def health():
    request_count.inc()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    request_count.inc()
    predict_count.inc()
    st = time.perf_counter()
    try:
        return predict_one(
            payload,
            db_path=app.state.db_path,
            model_path=app.state.model_path,
            meta_path=app.state.meta_path,
        )
    except Exception:
        predict_error_count.inc()
        raise
    finally:
        predict_latency_seconds.observe(time.perf_counter() - st)


@app.get("/metrics")
def metrics():
    request_count.inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
