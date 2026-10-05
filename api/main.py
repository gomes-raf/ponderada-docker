"""Backend de inferência: carrega o artefato model/model.joblib e serve predições.

Endpoints:
  GET  /health   -> verifica se o serviço está ativo e se o modelo foi carregado
  POST /predict  -> prevê o fechamento do BTC-USD no dia seguinte (D+1)
"""

import os
from contextlib import asynccontextmanager
from datetime import date, timedelta

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # O artefato chega pelo volume ./model -> /app/model (ver docker-compose.yml)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Artefato não encontrado em {MODEL_PATH}. Rode o treino antes.")
    artifact.update(joblib.load(MODEL_PATH))
    print(f"[api] modelo carregado de {MODEL_PATH} (treinado até {artifact['last_date']})")
    yield
    artifact.clear()


app = FastAPI(title="BTC Predictor", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    closes: list[float] | None = Field(
        default=None,
        description=(
            "Últimos fechamentos em US$, em ordem cronológica (mais antigo -> mais recente). "
            "Se omitido, usa os últimos fechamentos do dataset de treino."
        ),
        examples=[[83502.61, 83622.43, 83553.85, 84853.10, 84497.21, 84763.58, 86480.30]],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(artifact),
        "trained_until": artifact.get("last_date"),
    }


@app.post("/predict")
def predict(req: PredictRequest | None = None):
    n_lags = artifact["n_lags"]
    use_default = req is None or req.closes is None
    closes = artifact["last_window"] if use_default else req.closes

    if len(closes) != n_lags:
        raise HTTPException(422, f"Envie exatamente {n_lags} fechamentos; recebidos {len(closes)}.")
    if any(c <= 0 for c in closes):
        raise HTTPException(422, "Os fechamentos devem ser valores positivos.")

    prediction = float(artifact["model"].predict(np.asarray(closes, dtype=float).reshape(1, -1))[0])

    response = {
        "prediction_usd": round(prediction, 2),
        "horizon": "D+1",
        "input_closes": closes,
        "source": "dataset" if use_default else "request",
        "model": artifact["metrics"]["model"],
    }
    if use_default:
        last = date.fromisoformat(artifact["last_date"])
        response["predicted_date"] = (last + timedelta(days=1)).isoformat()
    return response
