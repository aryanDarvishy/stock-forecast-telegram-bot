import pandas as pd
import csv
from datetime import datetime
from pathlib import Path
from data import load_prices, make_summary, time_split
from metrics import rmse, mape
from models import get_models


LOG_PATH = Path("logs.csv")
TEST_SIZE = 30
FORECAST_HORIZON = 30

def process_text(text: str) -> dict:
    parts = text.split()
    pip = {"ok": False, "message": "..."}
    if len(parts) != 2:
        pip['message'] = ("Сообщение должно быть в формате: TICKER AMOUNT\n"
                          "Пример: AAPL 1000.")
        return pip
    ticker = parts[0].upper()
    try:
        amount = float(parts[1])
    except ValueError:
        pip['message'] = "Сумма должна быть числом"
        return pip
    if amount <= 0:
        pip['message'] = "Сумма должна быть больше 0"
        return pip
    return run_pipeline(ticker, amount)


def run_pipeline(ticker: str, amount: float) -> dict:
    scores = {}
    pipe = {
        "ok": True,
        "message": "",
        "ticker": ticker,
        "amount": amount,
        "last_price": None,
        "n_points": None,
    }

    df = load_prices(ticker)
    if df is None:
        pipe["ok"] = False
        pipe["message"] = "Не удалось получить данные по тикеру (проверь тикер или попробуй позже)."
        return pipe

    summary = make_summary(df)
    if summary is None:
        pipe["ok"] = False
        pipe["message"] = "Неожиданный формат данных (нет колонки Close)."
        return pipe

    pipe["last_price"], pipe["n_points"] = summary

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    split = time_split(close, TEST_SIZE)
    
    pipe['history_series'] = close
    
    if split is None:
        pipe['ok'] = False
        pipe['message'] = "Недостаточно данных для оценки (нужно больше 30 точек)."
        return pipe
    
    train, test = split
    h = len(test)

    models = get_models()

    for model in models:
        y_hat = model["predict"](train, h)
        scores[model["name"]] = {
            "rmse": rmse(test.values, y_hat),
            "mape": mape(test.values, y_hat),
        }

    best_name = min(scores, key=lambda name: scores[name]["rmse"])
    best_rmse = scores[best_name]["rmse"]
    best_mape = scores[best_name]["mape"]
    pipe["scores_by_models"] = scores
    
    best_model = next((model for model in models if model['name'] == best_name))
    future_hat = best_model['predict'](close, FORECAST_HORIZON)

    last_date = close.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=FORECAST_HORIZON + 1)[1:]
    forecast_last= float(future_hat[-1])
    last_price=float(pipe['last_price'])

    delta = forecast_last - last_price
    pct = delta / last_price * 100
    change_text = format_change(delta, pct)

    pipe["forecast_horizon"] = FORECAST_HORIZON
    pipe["forecast"] = future_hat.tolist() if hasattr(future_hat, "tolist") else list(future_hat)
    pipe["forecast_dates"] = list(future_dates)
    pipe["forecast_last"] = forecast_last 

    pipe["delta"] = float(delta)
    pipe["pct_change"] = float(pct)
    pipe["change_text"] = change_text
    pipe['model_name'] = best_name
    pipe['rmse_model'] = best_rmse
    pipe["mape_model"] = best_mape

    trade = simulate_extrema_trades(pipe["forecast"], pipe["amount"])
    pipe["trade"] = trade

    trades = trade["trades"]
    pipe["profit"] = trade["profit"]          
    pipe["profit_pct"] = trade["profit_pct"]  
    pipe["trades_text"] = format_trades_text(trades, pipe["forecast_dates"]) 

    pipe["message"] = format_message(pipe)
    
    return pipe  


def format_message(result: dict) -> str:
    return (
        f"Тикер: {result['ticker']}\n"
        f"Сумма: {result['amount']}\n"
        f"Скачано {result['n_points']} торговых дней.\n"
        f"Прогноз через 30 дней: {result['forecast_last']:.2f} $\n"
        f"{result['change_text']}\n"
        f"{result['trades_text']}\n"
        f"Лучшая модель: {result['model_name']}\n"
        f"RMSE: {result['rmse_model']:.2f}\n"
        f"MAPE: {result['mape_model']:.2f}%\n"
        f"Прибыль стратегии: {result['profit']:+.2f} $ ({result['profit_pct']:+.2f}%)\n"
    )


def log_result(user_id: int, result: dict) -> None:

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "ok": result.get("ok"),
        "ticker": result.get("ticker"),
        "amount": result.get("amount"),
        "last_price": result.get("last_price"),
        "n_points": result.get("n_points"),
        "model_name": result.get("model_name"),
        "rmse_model": result.get("rmse_model"),
        "mape_model": result.get("mape_model"),
        "forecast_last": result.get("forecast_last"),
        "delta": result.get("delta"),
        "pct_change": result.get("pct_change"),
        "profit": result.get("profit"),
        "profit_pct": result.get("profit_pct"),
        "trades_count": len(result.get("trade", {}).get("trades", [])),
        "message": result.get("message"),
    }
    fieldnames = list(row.keys())

    try:
        file_exists = LOG_PATH.exists()
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass



def format_change(delta, pct) -> str:
    if delta > 0:
        change_text = f"📈 Рост: {delta:+.2f} $ ({pct:+.2f}%)"
    elif delta < 0:
        change_text = f"📉 Падение: {delta:+.2f} $ ({pct:+.2f}%)"
    else:
        change_text = "➖ Без изменений"
    return change_text



def format_trades_text(trades: list[dict], forecast_dates, max_items: int = 6) -> str:
    if not trades:
        return "Сигналов buy/sell не найдено"

    lines = [f"Сделок: {len(trades)}"]

    for i, trade in enumerate(trades[:max_items], start=1):
        trade_date = forecast_dates[trade["index"]].strftime("%Y-%m-%d")
        lines.append(
            f"{i}) {trade['type']} — {trade_date}, "
            f"цена {trade['price']:.2f} $, "
            f"акций: {trade['shares']}"
        )

    if len(trades) > max_items:
        lines.append(f"... и ещё {len(trades) - max_items} сделок")

    return "\n".join(lines)



def find_local_extrema(prices: list[float]) -> dict:

    buys = list()
    sells = list()

    for i in range(1, len(prices)-1):
        if (prices[i] > prices[i-1]) and (prices[i] > prices[i+1]):
            sells.append(i)
        elif (prices[i] < prices[i-1]) and (prices[i] < prices[i+1]):
            buys.append(i)

    return {
        'buys': buys,
        'sells': sells
        
    }



def simulate_extrema_trades(prices, amount) -> dict:
    extrema = find_local_extrema(prices)

    buys = extrema["buys"]
    sells = extrema["sells"]

    buy_set = set(buys)
    sell_set = set(sells)

    cash = amount
    shares = 0
    trades = []

    for i, price in enumerate(prices):

        # BUY
        if i in buy_set and shares == 0:
            shares = int(cash // price)
            if shares > 0:
                cash -= shares * price
                trades.append({
                    "type": "BUY",
                    "index": i,
                    "price": price,
                    "shares": shares
                })

        # SELL
        elif i in sell_set and shares > 0:
            cash += shares * price
            trades.append({
                "type": "SELL",
                "index": i,
                "price": price,
                "shares": shares
            })
            shares = 0

    # Если в конце остались акции — считаем их по последней цене
    final_value = cash + shares * prices[-1]
    profit = final_value - amount
    profit_pct = round(profit / amount * 100, 2)
    return {
        "trades": trades,
        "final_value": final_value,
        "cash": cash,
        "shares": shares,
        "profit": profit,
        "profit_pct": profit_pct
    }