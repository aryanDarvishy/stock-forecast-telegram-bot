from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # pyright: ignore[reportMissingModuleSource]


def predict_naive_last(train: pd.Series, horizon: int) -> np.ndarray:
    last = float(train.iloc[-1])
    return np.full(horizon, last, dtype=float)


def predict_tf_gru(train: pd.Series, horizon: int, lookback: int = 30):
    values = np.asarray(pd.Series(train).dropna().values, dtype=float)

    if len(values) <= lookback + 5:
        last = float(values[-1])
        return np.full(horizon, last, dtype=float)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)

    X, y = make_windows(scaled, lookback)

    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])

    history = scaled.tolist()
    preds_scaled = []

    for _ in range(horizon):
        x_in = np.array(history[-lookback:], dtype=float).reshape(1, lookback, 1)
        y_hat = float(model.predict(x_in, verbose=0).reshape(-1)[0])
        preds_scaled.append(y_hat)
        history.append(y_hat)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1)
    return preds.astype(float)



def predict_ridge_lag(train: pd.Series, horizon: int, lag: int = 10) -> np.ndarray:
    values = np.asarray(train, dtype=float)

    if len(values) <= lag:
        last = float(values[-1])
        return np.full(horizon, last, dtype=float)

    X = []
    y = []

    for i in range(lag, len(values)):
        X.append(values[i - lag:i])
        y.append(values[i])

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    model = Ridge()
    model.fit(X, y)

    history = values.tolist()
    preds = []

    for _ in range(horizon):
        x_in = np.asarray(history[-lag:], dtype=float).reshape(1, -1)
        y_hat = float(model.predict(x_in)[0])

        prev = float(history[-1])

        max_step_pct = 0.03
        lower = prev * (1 - max_step_pct)
        upper = prev * (1 + max_step_pct)
        y_hat = float(np.clip(y_hat, lower, upper))

        preds.append(y_hat)
        history.append(y_hat)

    return np.asarray(preds, dtype=float)



def predict_arima(train: pd.Series, horizon: int) -> np.ndarray:
    values = np.asarray(train, dtype=float)

    if len(values) < 20:
        last = float(values[-1])
        return np.full(horizon, last, dtype=float)

    try:
        model = ARIMA(values, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)
    except Exception:
        last = float(values[-1])
        return np.full(horizon, last, dtype=float)



def make_windows(series: np.ndarray, lookback: int):
    series = np.asarray(series, dtype=float)

    X = []
    y = []

    for i in range(0, len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    X = X[..., None]

    return X, y



def get_models():
    return [
        {
            "name": "baseline_last",
            "predict": predict_naive_last,
        },
        {
            "name": "ridge_lag10",
            "predict": predict_ridge_lag,
        },
        {
            "name": "arima_111",
            "predict": predict_arima,
        },
        {
            "name": "tf_gru_lb30",
            "predict": predict_tf_gru,
        },
    ]