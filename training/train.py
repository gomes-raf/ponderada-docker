"""Treina um modelo para prever o fechamento do BTC-USD no dia seguinte.

Abordagem: regressão linear sobre os últimos N_LAGS fechamentos (janela deslizante).
Entrada:  data/btc.csv  (colunas: date, close, volume)
Saída:    model/model.joblib  (modelo + metadados)  e  model/metrics.json
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

DATA_PATH = os.getenv("DATA_PATH", "data/btc.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "model")
N_LAGS = int(os.getenv("N_LAGS", "7"))  # quantos fechamentos anteriores o modelo usa
TEST_FRACTION = 0.2  # últimos 20% da série ficam para teste


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df["close"].isna().any():
        raise ValueError("Há valores nulos na coluna 'close'.")
    return df


def make_windows(close: np.ndarray, n_lags: int):
    """X[i] = n_lags fechamentos em ordem cronológica; y[i] = fechamento do dia seguinte."""
    X = np.lib.stride_tricks.sliding_window_view(close[:-1], n_lags)
    y = close[n_lags:]
    return X, y


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),  # erro médio em US$
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),  # pune erros grandes
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 3),
    }


def direction_accuracy(y_true, y_pred, last_close) -> float:
    """% de dias em que o modelo acertou se o preço sobe ou cai em relação a hoje."""
    return round(float(np.mean((y_pred > last_close) == (y_true > last_close))) * 100, 1)


def main():
    df = load_data(DATA_PATH)
    close = df["close"].to_numpy(dtype=float)
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    print(f"[dados] {len(df)} dias, de {dates[0]} a {dates[-1]}")

    # 1) Preparação: janelas de N_LAGS dias -> alvo D+1
    X, y = make_windows(close, N_LAGS)
    target_dates = dates[N_LAGS:]

    # 2) Split cronológico (sem embaralhar)
    split = int(len(y) * (1 - TEST_FRACTION))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"[split] treino: {len(y_train)} amostras ({target_dates[0]} a {target_dates[split - 1]})")
    print(f"[split] teste:  {len(y_test)} amostras ({target_dates[split]} a {target_dates[-1]})")

    # 3) Treino e avaliação
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    last_close = X_test[:, -1]  # fechamento de "hoje" para cada alvo
    model_metrics = evaluate(y_test, y_pred)
    # Baseline ingênuo: "amanhã fecha igual a hoje"
    naive_metrics = evaluate(y_test, last_close)
    # Métrica principal de comparação: quanto o modelo reduz o MAE em relação ao ingênuo (>0 = melhor)
    skill = round((1 - model_metrics["mae"] / naive_metrics["mae"]) * 100, 2)
    model_metrics["direction_acc_pct"] = direction_accuracy(y_test, y_pred, last_close)
    up_days = round(float(np.mean(y_test > last_close)) * 100, 1)

    for name, m in (("modelo:  ", model_metrics), ("ingênuo: ", naive_metrics)):
        print(f"[teste] {name} MAE = US$ {m['mae']:,.2f} | RMSE = US$ {m['rmse']:,.2f} | MAPE = {m['mape_pct']}%")
    print(f"[teste] ganho sobre o ingênuo (MAE): {skill:+.2f}%  (>0 = modelo melhor)")
    print(f"[teste] acerto de direção: {model_metrics['direction_acc_pct']}%  "
          f"(referência: 50%; dias de alta no teste: {up_days}%)")

    # 4) Re-treina com a série inteira para usar os dados mais recentes na inferência
    final_model = LinearRegression().fit(X, y)

    # 5) Exporta o artefato
    os.makedirs(MODEL_DIR, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {
        "model": "LinearRegression",
        "n_lags": N_LAGS,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_period": [str(target_dates[split]), str(target_dates[-1])],
        "test_model": model_metrics,
        "test_naive_baseline": naive_metrics,
        "skill_vs_naive_mae_pct": skill,
        "test_up_days_pct": up_days,
        "trained_at": trained_at,
    }
    artifact = {
        "model": final_model,
        "n_lags": N_LAGS,
        "last_date": str(dates[-1]),
        "last_window": close[-N_LAGS:].tolist(),  # usado se o cliente não enviar fechamentos
        "metrics": metrics,
    }
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(artifact, model_path)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    next_close = float(final_model.predict([close[-N_LAGS:]])[0])
    print(f"[artefato] salvo em {model_path}")
    print(f"[exemplo] previsão para o dia seguinte a {dates[-1]}: US$ {next_close:,.2f}")


if __name__ == "__main__":
    main()
