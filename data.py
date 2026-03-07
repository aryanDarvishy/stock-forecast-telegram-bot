import pandas as pd
import yfinance as yf


def load_prices(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="2y", progress=False)
    except Exception as e:
        return None
    if df.empty:
        return None
    return df


def make_summary(df) -> tuple[float, int] | None:
    if 'Close' not in df.columns:
        return None
    last = df["Close"].iloc[-1]
    if isinstance(last, pd.Series):
        last = last.iloc[0]
    last_price = round(float(last), 2)
    size =  len(df)
    return (last_price, size) 


def time_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series] | None:
    if len(series) <= test_size:
        return None
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test
