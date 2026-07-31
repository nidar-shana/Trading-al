# Trading-algo
This is a Trading Algo project
# Trading-algo
This is a Trading Algo project.

## Stock Predictor

A simple stock price predictor using historical closing prices and a linear regression model.

### Files

- `stock_predictor.py` — downloads stock data from Yahoo Finance, trains a linear regression model, and predicts the next closing price.
- `requirements.txt` — Python package dependencies.

### Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the predictor:

```bash
python stock_predictor.py --symbol AAPL --period 1y --interval 1d --lookback 5 --test
```

3. Example output:

```bash
Downloading AAPL data for period=1y, interval=1d...
Predicted next closing price for AAPL: $XXX.XX
Test MSE: 0.XXXX
Test R^2: 0.XXXX
```
