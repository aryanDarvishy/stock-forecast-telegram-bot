from io import BytesIO
import matplotlib
matplotlib.use("Agg")  # важно: ДО pyplot
import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(result: dict)-> BytesIO:
    history : pd.Series = result['history_series'].tail(130)
    forecast_dates = result['forecast_dates']
    forecast = result['forecast']

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(history.index, history.values, label="History", linewidth=2)
    ax.plot(forecast_dates, forecast, label="Forecast", linewidth=2)

    ax.set_title(f"{result['ticker']} Forecast (model: {result['model_name']})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf