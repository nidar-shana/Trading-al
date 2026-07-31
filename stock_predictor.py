import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def download_stock_data(symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    if data.empty:
        raise ValueError(
            f'No data found for symbol {symbol} with period={period} and interval={interval}.'
        )
    return data


def prepare_features(data: pd.DataFrame, lookback: int = 5):
    closes = data['Close'].astype(float).reset_index(drop=True)
    if len(closes) <= lookback:
        raise ValueError('Not enough data to build features. Increase the period or decrease lookback.')

    X = []
    y = []
    for i in range(lookback, len(closes)):
        X.append(closes[i - lookback:i].values)
        y.append(closes[i])

    return np.asarray(X), np.asarray(y)


def train_model(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, scaler, mse, r2


def predict_next_price(model: LinearRegression, scaler: StandardScaler, recent_prices: np.ndarray) -> float:
    recent_scaled = scaler.transform(recent_prices.reshape(1, -1))
    return float(model.predict(recent_scaled)[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple stock price predictor using historical closing prices.'
    )
    parser.add_argument('--symbol', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', default='1y', help='History period, e.g. 1y, 6mo, 2y')
    parser.add_argument('--interval', default='1d', help='Data interval, e.g. 1d, 1wk')
    parser.add_argument('--lookback', type=int, default=5, help='Number of previous days to use as features')
    parser.add_argument('--test', action='store_true', help='Show evaluation metrics')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'Downloading {args.symbol} data for period={args.period}, interval={args.interval}...')
    try:
        data = download_stock_data(args.symbol, args.period, args.interval)
    except Exception as exc:
        print(f'Error downloading data: {exc}', file=sys.stderr)
        sys.exit(1)

    X, y = prepare_features(data, args.lookback)
    model, scaler, mse, r2 = train_model(X, y)

    recent_prices = X[-1]
    predicted_price = predict_next_price(model, scaler, recent_prices)

    print(f'Predicted next closing price for {args.symbol}: ${predicted_price:.2f}')
    if args.test:
        print(f'Test MSE: {mse:.4f}')
        print(f'Test R^2: {r2:.4f}')


if __name__ == '__main__':
    main()
