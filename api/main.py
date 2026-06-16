import os
import time
import logging
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# 1. Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# 2. Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 3. FastAPI Init
app = FastAPI(title='Predictive Maintenance API', version='1.0.0')

metrics = {'total_requests': 0, 'predictions': [], 'latencies': [], 'failures_predicted': 0, 'errors': 0}

# 4. Load Model
MODEL_NAME = 'PredictiveMaintenance'
model = None

try:
    versions = client.get_latest_versions(MODEL_NAME)
    if versions:
        latest_version = versions[0]
        model_uri = f"runs:/{latest_version.run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded: {latest_version.version}")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# 5. Schema
class PredictionRequest(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    age_days: int

# 6. Predict Endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    try:
        start_time = time.time()
        input_data = pd.DataFrame([request.dict()])
        prob = model.predict(input_data)[0]
        will_fail = bool(prob >= 0.5)
        latency = (time.time() - start_time) * 1000
        
        metrics['total_requests'] += 1
        if will_fail: metrics['failures_predicted'] += 1
        
        return {
            "will_fail": will_fail,
            "probability": float(prob),
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        metrics['errors'] += 1
        raise HTTPException(status_code=500, detail=str(e))