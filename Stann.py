import os
import datetime
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.base import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import talib
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import joblib
import urllib3
import json
import tensorflow as tf

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
api_key_alpha_vantage = ""
api_key_finhub = ""
rapidapi_key = ""
alpha_vantage_base_url = "https://www.alphavantage.co/query"
finhub_base_url = "https://finnhub.io/api/v1"
macrotrends_base_url = "https://macrotrends.net"

# Stock prediction model parameters
look_back = 60
epochs = 50
batch_size = 32

scaler = MinMaxScaler(feature_range=(0, 1))
cache_folder = "/home/jesse/newstann/NewStannCacheData"

# Update price range for stock filtering
min_price = 1.01
max_price = 8.01


def get_stock_data_from_yahoo(ticker: str, start_date: str, end_date: str, period: str = 'max') -> pd.DataFrame:
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, period=period)
        if stock_data.empty:
            return None
        return stock_data
    except ValueError:
        if period != '1d' and period != '5d':
            print(f"Failed to retrieve stock data for {ticker} with period '{period}'. Trying '1d'...")
            return get_stock_data_from_yahoo(ticker, start_date, end_date, period='1d')
        else:
            print(f"Failed to retrieve stock data for {ticker}. Skipping...")
            return None


def get_stock_data_with_cache(ticker: str, start_date: str, end_date: str, period: str = 'max') -> pd.DataFrame:
    cache_filename = os.path.join(cache_folder, f"stock_data_cache_{ticker}_{start_date}_{end_date}_{period}.pkl")
    try:
        stock_data = joblib.load(cache_filename)
        print(f"Retrieved stock data for {ticker} from cache.")
        return stock_data
    except FileNotFoundError:
        print(f"Cache file for {ticker} not found. Fetching stock data from Yahoo Finance...")
        stock_data = get_stock_data_from_yahoo(ticker, start_date, end_date, period)
        if stock_data is not None:
            os.makedirs(cache_folder, exist_ok=True)
            joblib.dump(stock_data, cache_filename)
            print(f"Stock data for {ticker} cached successfully.")
        return stock_data


def delete_cache_files():
    for filename in os.listdir(cache_folder):
        file_path = os.path.join(cache_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error while deleting cache file: {str(e)}")


def get_stock_data_from_macrotrends(ticker: str, range: str = "1y") -> pd.DataFrame:
    url = f"{macrotrends_base_url}/stocks/charts/{ticker}/{ticker}/stock-price-history"
    params = {
        "timeframe": range,
        "end_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        script_tag = soup.find("script", {"id": "chartdata"})
        if not script_tag:
            return None

        data = json.loads(script_tag.string)
        stock_data = pd.DataFrame(data)

        if "date" in stock_data.columns:
            stock_data.set_index("date", inplace=True)
        else:
            print(f"'date' column not found in the data for {ticker}.")
            return None

        stock_data.index = pd.to_datetime(stock_data.index)
        return stock_data
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve stock data for {ticker}. Error: {str(e)}")
        return None


def get_stock_data_from_alpha_vantage(ticker: str, start_date: str, end_date: str, function: str) -> pd.DataFrame:
    url = alpha_vantage_base_url
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": api_key_alpha_vantage,
        "outputsize": "full",
        "datatype": "json"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "Time Series (Daily)" not in data:
            print(f"No stock data found for {ticker} in Alpha Vantage response.")
            return None

        stock_data = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        stock_data.columns = ["Open", "High", "Low", "Close", "Volume"]
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.sort_index(ascending=True, inplace=True)

        if start_date:
            stock_data = stock_data[stock_data.index >= start_date]
        if end_date:
            stock_data = stock_data[stock_data.index <= end_date]

        return stock_data
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve stock data for {ticker} from Alpha Vantage. Error: {str(e)}")
        return None


def get_stock_data_from_twelve_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": rapidapi_key
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "values" not in data:
            print(f"No stock data found for {ticker} in Twelve Data response.")
            return None

        stock_data = pd.DataFrame(data["values"])
        stock_data.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"])
        stock_data.set_index("Datetime", inplace=True)
        stock_data.sort_index(ascending=True, inplace=True)

        return stock_data
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve stock data for {ticker} from Twelve Data. Error: {str(e)}")
        return None


def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock_data_yahoo = get_stock_data_with_cache(ticker, start_date, end_date)
    stock_data_macrotrends = get_stock_data_from_macrotrends(ticker)

    if stock_data_yahoo is None and stock_data_macrotrends is None:
        stock_data_alpha_vantage = get_stock_data_from_alpha_vantage(ticker, start_date, end_date, "TIME_SERIES_DAILY")
        stock_data_twelve_data = get_stock_data_from_twelve_data(ticker, start_date, end_date)

        if stock_data_alpha_vantage is not None:
            return stock_data_alpha_vantage
        elif stock_data_twelve_data is not None:
            return stock_data_twelve_data
        else:
            return None
    elif stock_data_yahoo is None:
        return stock_data_macrotrends
    elif stock_data_macrotrends is None:
        return stock_data_yahoo
    else:
        stock_data = pd.concat([stock_data_yahoo, stock_data_macrotrends]).sort_index()
        stock_data = stock_data[~stock_data.index.duplicated(keep="first")]
        return stock_data


def get_tickers_from_yahoo() -> list[str]:
    url = "https://finance.yahoo.com/most-active"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        tickers = []

        for a in soup.find_all("a", {"class": "Fw(600) C($linkColor)"}):
            ticker = a.text.strip()
            if ticker:
                tickers.append(ticker)
        return tickers
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve tickers from Yahoo Finance. Error: {str(e)}")
        return []


def get_tickers_from_nasdaq() -> list[str]:
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    }
    try:
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)

        tickers = [row["symbol"] for row in data["data"]["table"]["rows"]]
        return tickers
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve tickers from Nasdaq. Error: {str(e)}")
        return []


def get_tickers_from_sources() -> list[str]:
    tickers_yahoo = get_tickers_from_yahoo()
    tickers_nasdaq = get_tickers_from_nasdaq()

    return list(set(tickers_yahoo + tickers_nasdaq))


def calculate_bollinger_bands(stock_data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    stock_data["Middle Band"] = stock_data["Close"].rolling(window=window).mean()
    stock_data["Upper Band"] = stock_data["Middle Band"] + (stock_data["Close"].rolling(window=window).std() * num_std)
    stock_data["Lower Band"] = stock_data["Middle Band"] - (stock_data["Close"].rolling(window=window).std() * num_std)
    return stock_data


def calculate_future_price(stock_data: pd.DataFrame, days: int = 30) -> pd.Series:
    future_prices = stock_data['Close'].shift(-days)
    return future_prices


def calculate_roi(stock_data: pd.DataFrame, days: int = 30) -> pd.Series:
    future_prices = calculate_future_price(stock_data, days)
    roi = (future_prices - stock_data['Close']) / stock_data['Close']
    return roi


def preprocess_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

    stock_data['SMA_5'] = talib.SMA(stock_data['Close'], timeperiod=5)
    stock_data['SMA_10'] = talib.SMA(stock_data['Close'], timeperiod=10)
    stock_data['SMA_20'] = talib.SMA(stock_data['Close'], timeperiod=20)
    stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
    stock_data['SMA_100'] = talib.SMA(stock_data['Close'], timeperiod=100)
    stock_data['SMA_200'] = talib.SMA(stock_data['Close'], timeperiod=200)

    stock_data['EMA_5'] = talib.EMA(stock_data['Close'], timeperiod=5)
    stock_data['EMA_10'] = talib.EMA(stock_data['Close'], timeperiod=10)
    stock_data['EMA_20'] = talib.EMA(stock_data['Close'], timeperiod=20)
    stock_data['EMA_50'] = talib.EMA(stock_data['Close'], timeperiod=50)
    stock_data['EMA_100'] = talib.EMA(stock_data['Close'], timeperiod=100)
    stock_data['EMA_200'] = talib.EMA(stock_data['Close'], timeperiod=200)

    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
    stock_data['MACD'], stock_data['MACD_SIGNAL'], stock_data['MACD_HIST'] = talib.MACD(stock_data['Close'])

    stock_data = calculate_bollinger_bands(stock_data)
    stock_data['ROCP'] = talib.ROCP(stock_data['Close'], timeperiod=10)

    stock_data.dropna(inplace=True)

    return stock_data


def split_data(stock_data: pd.DataFrame, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_size = int(len(stock_data) * (1 - test_size))
    train_data = stock_data.iloc[:train_size]
    test_data = stock_data.iloc[train_size:]

    x_train = train_data.drop(['Close'], axis=1).values
    y_train = train_data['Close'].values

    x_test = test_data.drop(['Close'], axis=1).values
    y_test = test_data['Close'].values

    return x_train, y_train, x_test, y_test


def train_lstm_model(x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int) -> Sequential:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def train_svm_model(x_train: np.ndarray, y_train: np.ndarray) -> SVR:
    svm_model = SVR(kernel='linear')
    svm_model.fit(x_train, y_train)

    return svm_model


def predict_lstm_model(stock_data: pd.DataFrame) -> np.ndarray:
    processed_data = preprocess_data(stock_data)
    scaled_data = scaler.fit_transform(processed_data)
    x_train, y_train, x_test, y_test = split_data(scaled_data)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = train_lstm_model(x_train, y_train, epochs, batch_size)
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices.flatten()


def predict_svm_model(stock_data: pd.DataFrame) -> np.ndarray:
    processed_data = preprocess_data(stock_data)
    scaled_data = scaler.fit_transform(processed_data)
    x_train, y_train, x_test, y_test = split_data(scaled_data)

    model = train_svm_model(x_train, y_train)
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

    return predicted_prices.flatten()


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    evaluation = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

    return evaluation


def main():
    tickers = get_tickers_from_sources()
    if len(tickers) == 0:
        print("No tickers found. Exiting...")
        return

    results = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        # Get the stock data
        start_date = "2023-01-01"
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        stock_data = get_stock_data(ticker, start_date, end_date)

        if stock_data is None:
            print(f"Failed to retrieve stock data for {ticker}. Skipping...")
            continue

        # Calculate Bollinger Bands
        stock_data = calculate_bollinger_bands(stock_data)

        # Get the current share price
        current_price = stock_data['Close'].iloc[-1]
        print(f"Current Share Price: {current_price:.2f}")

        # Predict the stock price using LSTM
        predicted_price_lstm = predict_lstm_model(stock_data)
        if predicted_price_lstm is not None:
            print(f"Predicted Stock Price (LSTM): {predicted_price_lstm:.2f}")
        else:
            print("Insufficient data for LSTM prediction.")

        # Predict the stock price using SVM
        predicted_price_svm = predict_svm_model(stock_data)
        if predicted_price_svm is not None:
            print(f"Predicted Stock Price (SVM): {predicted_price_svm:.2f}")
        else:
            print("Insufficient data for SVM prediction.")

        # Calculate ROI
        roi = calculate_roi(stock_data)
        if roi is not None:
            print(f"ROI: {roi:.2f}%")

            # Append current price, ROI, and predictions to the results list
            if predicted_price_lstm is None:
                results.append((ticker, current_price, roi, predicted_price_svm))
            else:
                results.append((ticker, current_price, roi, predicted_price_lstm, predicted_price_svm))

    # Display the top 10 stocks with the best ROI
    print("\nTop 10 Stocks with the Best ROI:")
    results = sorted(results, key=lambda x: x[2], reverse=True)  # Sort based on ROI (index 2)
    for i in range(min(10, len(results))):
        ticker, current_price, roi, *predictions = results[i]
        prediction_values = ', '.join([f"{p:.2f}" for p in predictions])
        print(f"{ticker}: Current Price: {current_price:.2f}, ROI: {roi:.2f}%, Predictions: {prediction_values}")


if __name__ == "__main__":
    main()
