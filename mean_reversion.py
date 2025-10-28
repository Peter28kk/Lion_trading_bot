import ccxt
import pandas as pd
import time
import logging
import asyncio
import websockets
import os
import json
import ta
import requests
import hashlib
import logging
import hmac
import base64
from ta.volatility import AverageTrueRange
import numpy as np
from datetime import datetime, timezone
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import welch, find_peaks
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import warnings
try:
    from scipy.sparse import SparseEfficiencyWarning
except Exception:
    # Fallback: define a lightweight warning class if SciPy is not available
    class SparseEfficiencyWarning(Warning):
        pass

latest_price = None
prices = []
order_history = pd.DataFrame({
    'timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
    'symbol': pd.Series(dtype='str'),
    'side': pd.Series(dtype='str')
})

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pd.set_option('display.float_format', '{:.10f}'.format)
pd.set_option('display.max_columns', None)
global last_processed_timestamp, last_processed_minute
last_processed_minute = None  # Last processed minute
last_processed_timestamp = None
last_processed_index = None  # Last processed index

df = pd.DataFrame()  # Global DataFrame for OHLCV data

def process_new_candle(new_candle):
    global df, last_processed_timestamp, prices

    if not isinstance(df, pd.DataFrame):
       logging.error(f"`df` is corrupted. Type: {type(df)}. Resetting to empty DataFrame.")

    try:
        new_row = pd.DataFrame([{
            'timestamp': new_candle['timestamp'],
            'open': new_candle['open'],
            'high': new_candle['high'],
            'low': new_candle['low'],
            'close': new_candle['close'],
            'volume': new_candle['volume'],
            'fast_SMA': np.nan,
            'medium_SMA': np.nan,
            'slow_SMA': np.nan,
            'RSI': np.nan,
            'VO_diff_pct': np.nan,
            'signal': 0,
            'buy_price': np.nan,
            'sell_price': np.nan,
            'trend': '',
            'decision': '',
            'med_cross': '',
            'slow_cross': ''
        }])

        logging.debug(f"df type before concat: {type(df)}")

        df = pd.concat([df, new_row], ignore_index=True).tail(500).reset_index(drop=True)

        
        df = calculate_indicators(df)

        if last_processed_timestamp is None:
            last_processed_timestamp = pd.Timestamp.min.tz_localize('UTC')

        df_to_process = df[df['timestamp'] > last_processed_timestamp]
        if df_to_process.empty:
            return df, "No new data", 0
        
        start_idx = df.index.get_loc(df_to_process.index[0])
        if start_idx == 0:
          start_idx = 1

        
        df, decision = generate_signals(df, start_idx=1, 
                          account_equity=100000, 
                          risk_per_trade=0.02,
                          atr_multiplier=3.5,
                          adx_threshold=20,
                          ema_fast=50,
                          ema_slow=200,
                          momentum_window=14)

        last_processed_timestamp = df.iloc[-1]['timestamp']
        latest_row = df.iloc[-1]
        decision = latest_row.get('decision', 'N/A')
        signal = int(latest_row.get('signal', 0))

        logging.info(f"decision: {decision} | signal: {signal}")
        return df, decision, signal

    except Exception as e:
        logging.error(f"Error processing new candle: {e}", exc_info=True)
        return df, "Error", 0

def get_api_keys():
    api_key = os.getenv('OKX_API_KEY')
    api_secret = os.getenv('OKX_API_SECRET')
    api_passphrase = os.getenv('OKX_API_PASSPHRASE')
    
    if all([api_key, api_secret, api_passphrase]):
        logging.info("API keys loaded from environment variables.")
        return api_key, api_secret, api_passphrase
    else:
        logging.error("API keys not found in environment variables.")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config.get('api_key')
        api_secret = config.get('api_secret')
        api_passphrase = config.get('api_passphrase')
        if all([api_key, api_secret, api_passphrase]):
            logging.info("API keys loaded from config.json.")
            return api_key, api_secret, api_passphrase
    except FileNotFoundError:
        logging.warning("config.json not found. Proceeding to user input.")
    
    logging.info("Please enter your OKX API credentials:")
    api_key = input("API Key: ")
    api_secret = input("API Secret: ")
    api_passphrase = input("API Passphrase: ")
    
    if not all([api_key, api_secret, api_passphrase]):
        raise ValueError("All API credentials are required.")
    
    with open('config.json', 'w') as f:
        json.dump({
            'api_key': api_key,
            'api_secret': api_secret,
            'api_passphrase': api_passphrase
        }, f)
    logging.info("API keys saved to config.json for future use.")
    return api_key, api_secret, api_passphrase

API_KEY, API_SECRET, API_PASSPHRASE = get_api_keys()

# Initialize ccxt for live trading (sandbox disabled)
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSPHRASE,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
    'sandbox': False,  # Live trading mode
})

# Global settings
SYMBOL = 'PEPE-USDT'
TIMEFRAME = '1m'  # 1-minute candles
TRADE_NOTIONAL = 1.0  # Trade amount in USD (e.g., 50 cents per trade)

def custom_request(endpoint, method='GET', params=None, retries=5, timeout=500):
    url = 'https://www.okx.com/api/v5' + endpoint
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    request_path = f'/api/v5{endpoint}'
    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-PASSPHRASE': API_PASSPHRASE,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'x-simulated-trading': '0',  # Live trading
        'Content-Type': 'application/json'
    }
    
    body = json.dumps(params) if params else ''
    message = timestamp + method.upper() + request_path + body
    signature = hmac.new(API_SECRET.encode('utf-8'),
                         message.encode('utf-8'),
                         hashlib.sha256).digest()
    headers['OK-ACCESS-SIGN'] = base64.b64encode(signature).decode('utf-8')
    
    for attempt in range(retries):
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=body, timeout=timeout)
    
            logging.info(f"API Response (Status Code: {response.status_code}): {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Request failed with status code {response.status_code}: {response.text}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(5)
    raise Exception("Failed to execute the API request after multiple retries.")

def fetch_ohlcv(symbol, timeframe, limit=500):
    endpoint = '/market/candles'
    params = {
        'instId': symbol,
        'bar': timeframe,
        'limit': limit
    }

    try:
        ohlcv_data = custom_request(endpoint, method='GET', params=params)
        if 'data' in ohlcv_data and ohlcv_data['data']:
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'quote_volume', 'number_of_trades', 'realized_price'
            ]
            df_full = pd.DataFrame(ohlcv_data['data'], columns=columns)
            df_full = df_full.iloc[::-1].reset_index(drop=True)

            df_full['timestamp'] = pd.to_datetime(
                pd.to_numeric(df_full['timestamp'], errors='coerce'), unit='ms', utc=True
            )
            float_cols = ['open', 'high', 'low', 'close', 'volume']
            df_full[float_cols] = df_full[float_cols].astype(float)

            df = df_full[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            logging.info("Fetched clean OHLCV data:")
            logging.info(df.tail())

            return df
        else:
            logging.warning("No OHLCV data found.")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Failed to fetch OHLCV data: {e}")
        return pd.DataFrame()

def detect_dominant_cycle(prices, max_lag=100):
    """
    Detect market cycles using spectral analysis with:
    1. HP Filter detrending
    2. Welch periodogram
    3. Peak detection with harmonic validation
    """
    try:
        # 1. Clean and validate data
        prices_clean = prices[~np.isnan(prices)]
        min_data_points = 3 * max_lag
        if len(prices_clean) < min_data_points:
            print(f"Warning: Need at least {min_data_points} data points")
            return 30, 15

        # 2. HP Filter detrending (λ=14400 for daily data)
        cycle_component, _ = hpfilter(prices_clean, lamb=14400)
        
        # 3. Welch periodogram (better than raw FFT for noisy data)
        fs = 1.0  # Sampling frequency (1 observation per period)
        nperseg = min(256, len(cycle_component)//4)
        freqs, power = welch(cycle_component, fs=fs, nperseg=nperseg)
        
        # 4. Convert to periods and filter plausible ranges
        periods = 1 / freqs[1:]  # Skip infinite period at freq=0
        power = power[1:]
        valid_mask = (periods >= 10) & (periods <= max_lag)
        periods = periods[valid_mask]
        power = power[valid_mask]
        
        if len(periods) == 0:
            return 30, 15

        # 5. Find significant peaks
        min_peak_height = 0.5 * np.max(power)
        peaks, properties = find_peaks(power, 
                                     height=min_peak_height,
                                     prominence=0.3*np.max(power))
        
        if len(peaks) == 0:
            return 30, 15

        # 6. Select dominant cycle (prioritizing strongest prominence)
        dominant_idx = peaks[np.argmax(properties['prominences'])]
        dominant_period = int(round(periods[dominant_idx]))
        
        # 7. Harmonic validation (avoid picking harmonics)
        for period in periods[peaks]:
            if 0.45*dominant_period < period < 0.55*dominant_period:
                dominant_period = int(round(period*2))  # Found 1/2 harmonic
            
        return dominant_period, dominant_period//2
        
    except Exception as e:
        print(f"Cycle detection error: {e}")
        return 30, 15

def calculate_ATR(df, cycle_length):
    """Always return DataFrame with 'ATR' column"""
    try:
        window = max(int(cycle_length), 14)
        df['ATR'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=window
        ).average_true_range()
    except Exception as e:
        logging.error(f"ATR calculation failed: {e}")
        df['ATR'] = np.nan  # Create column even if calculation fails
    
    return df

def calculate_indicators(df):
    """Guarantee 'sma' and 'ATR' columns exist"""
    try:
        prices = df['close'].dropna().values
        
        # Get cycle lengths with integer fallback
        full_cycle, half_cycle = detect_dominant_cycle(prices)
        full_cycle = int(full_cycle) if not np.isnan(full_cycle) else 30
        half_cycle = int(half_cycle) if not np.isnan(half_cycle) else 15

        print(f"Full cycle: {full_cycle}, Half cycle: {half_cycle}")
        
        # Calculate SMA
        df['sma'] = df['close'].rolling(
            window=max(full_cycle, 10),
            min_periods=1
        ).mean()
        
        # Calculate ATR
        df = calculate_ATR(df, half_cycle)

        df['RSI'] = RSIIndicator(df['close'], window=10).rsi()
        
    except Exception as e:
        logging.error(f"Indicator calculation failed: {e}")
        # Ensure columns exist even on failure
        df['sma'] = df['close'].copy()
        df['ATR'] = 0.0
    
    return df

from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def generate_signals(df, start_idx=1,
                    account_equity=100000, 
                    risk_per_trade=0.02,
                    atr_multiplier=3.5,
                    adx_threshold=20,
                    ema_fast=50,
                    ema_slow=200,
                    momentum_window=14):
    """
    Elite Trend Following Strategy (1M/5M Timeframe)
    Signals: 
        1 = Buy (Enter Long or Exit Short)
        -1 = Sell (Enter Short or Exit Long)
        0 = No Action
    Returns: (df, decision)
    """
    import numpy as np
    from ta.volatility import AverageTrueRange
    from ta.trend import EMAIndicator, ADXIndicator, MACD
    from ta.momentum import RSIIndicator
    
    # =====================
    # 1. Initialize DataFrame
    # =====================
    df = df.copy()
    # Initialize all columns properly
    for col in ['signal', 'decision', 'entry_price', 'stop_loss', 'trailing_stop']:
        if col not in df.columns:
            if col == 'decision':
                df[col] = ''
            elif col == 'signal':
                df[col] = 0  # Default to no action
            else:
                df[col] = np.nan

    # =====================
    # 2. Core Indicators
    # =====================
    # Volatility System
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], 
                               close=df['close'], window=14).average_true_range()
    
    # Trend Direction System
    df['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], 
                           close=df['close'], window=50).adx()
    
    # Momentum Confirmation
    macd = MACD(close=df['close'], window_fast=12, 
              window_slow=26, window_sign=9)
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # Volume Flow
    df['volume_ma'] = df['volume'].rolling(20).mean().replace(0, 1)
    df['vroc'] = df['volume'] / df['volume_ma']
    
    # =====================
    # 3. Trading System
    # =====================
    active_trade = False
    trade_direction = None  # 'long' or 'short'
    decision = 'No trades'
    max_profit = 0
    entry_price = np.nan
    trailing_stop = np.nan
    entry_bar = 0
    
    for i in range(start_idx, len(df)):
        try:
            # Current values
            c = df['close'].iloc[i]
            h = df['high'].iloc[i]
            l = df['low'].iloc[i]
            atr = df['ATR'].iloc[i]
            ema_fast_val = df['ema_fast'].iloc[i]
            prev_ema_fast_val = df['ema_fast'].iloc[i-1] if i > 0 else np.nan
            prev_ema_slow_val = df['ema_slow'].iloc[i-1] if i > 0 else np.nan
            ema_slow_val = df['ema_slow'].iloc[i]
            macd_line = df['MACD_line'].iloc[i]
            macd_signal = df['MACD_signal'].iloc[i]
            rsi = df['RSI'].iloc[i]
            vroc = min(df['vroc'].iloc[i], 5)  # Cap outliers at 5x
            adx_val = df['ADX'].iloc[i]
            
            # Default decision for this bar
            df.at[i, 'decision'] = "No trade"
            
            # ===== ENTRY LOGIC =====
            if not active_trade and adx_val >= adx_threshold:
                # Reset signal for this bar
                df.at[i, 'signal'] = 0
                
                # Bullish convergence (Long entry)
                if (prev_ema_fast_val < prev_ema_slow_val and c > ema_fast_val > ema_slow_val and
                    macd_line > macd_signal):
                    
                    active_trade = True
                    trade_direction = 'long'
                    entry_price = c
                    initial_sl = c - atr_multiplier * atr
                    trailing_stop = initial_sl
                    max_profit = 0
                    entry_bar = i
                    
                    # Generate buy signal (1)
                    df.at[i, 'signal'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss'] = initial_sl
                    df.at[i, 'trailing_stop'] = initial_sl
                    decision = f"ENTER LONG @ {c:.5f} | SL={initial_sl:.5f}"
                    df.at[i, 'decision'] = decision
                    
                # Bearish convergence (Short entry)
                elif (prev_ema_fast_val > prev_ema_slow_val and c < ema_fast_val < ema_slow_val and
                      macd_line < macd_signal):
                    
                    active_trade = True
                    trade_direction = 'short'
                    entry_price = c
                    initial_sl = c + atr_multiplier * atr
                    trailing_stop = initial_sl
                    max_profit = 0
                    entry_bar = i
                    
                    # Generate sell signal (-1)
                    df.at[i, 'signal'] = -1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss'] = initial_sl
                    df.at[i, 'trailing_stop'] = initial_sl
                    decision = f"ENTER SHORT @ {c:.5f} | SL={initial_sl:.5f}"
                    df.at[i, 'decision'] = decision

            # ===== EXIT & TRAILING MANAGEMENT =====
            elif active_trade:
                # ---- LONG POSITION ----
                if trade_direction == 'long':
                    # Calculate current profit ratio
                    current_profit = c - entry_price
                    profit_ratio = current_profit / atr
                    max_profit = max(max_profit, profit_ratio)
                    
                    # Adaptive Trailing Stop
                    if max_profit > 3.0:
                        new_trail = h - 1.8 * atr
                    elif max_profit > 1.5:
                        new_trail = h - 2.0 * atr
                    else:
                        new_trail = max(entry_price - 0.5 * atr, trailing_stop)
                    
                    trailing_stop = max(trailing_stop, new_trail)
                    
                    # Exit conditions (independent of ADX)
                    exit_conditions = [
                        l <= trailing_stop,  # Stop loss hit
                        macd_line < macd_signal,  # Momentum reversal
                        c < ema_fast_val,  # Price below fast EMA
                        profit_ratio < -1.5  # Emergency stop
                    ]
                    
                    if any(exit_conditions):
                        exit_price = min(l, trailing_stop)
                        reason = "Stop hit" if exit_conditions[0] else \
                                 "MACD reversal" if exit_conditions[1] else \
                                 "EMA break" if exit_conditions[2] else \
                                 "Time limit" if exit_conditions[3] else "Emergency stop"
                        
                        # Generate sell signal (-1) to exit long position
                        df.at[i, 'signal'] = -1
                        decision = (f"EXIT LONG @ {exit_price:.5f} | "
                                    f"Profit: {exit_price-entry_price:.5f} ({reason})")
                        df.at[i, 'decision'] = decision
                        df.at[i, 'stop_loss'] = trailing_stop
                        df.at[i, 'trailing_stop'] = trailing_stop
                        
                        # Reset trade state
                        active_trade = False
                        trade_direction = None
                    else:
                        # Maintain long position (no signal change)
                        df.at[i, 'signal'] = 0  # No new action needed
                        df.at[i, 'stop_loss'] = trailing_stop
                        df.at[i, 'trailing_stop'] = trailing_stop
                        df.at[i, 'decision'] = f"Holding LONG | Current Profit: {current_profit:.5f}"
                
                # ---- SHORT POSITION ----
                elif trade_direction == 'short':
                    current_profit = entry_price - c
                    profit_ratio = current_profit / atr
                    max_profit = min(max_profit, -profit_ratio)
                    
                    # Adaptive Trailing Stop
                    if max_profit < -3.0:
                        new_trail = l + 1.8 * atr
                    elif max_profit < -1.5:
                        new_trail = l + 2.0 * atr
                    else:
                        new_trail = min(entry_price + 0.5 * atr, trailing_stop)
                    
                    trailing_stop = min(trailing_stop, new_trail)
                    
                    # Exit conditions
                    exit_conditions = [
                        h >= trailing_stop,  # Stop loss hit
                        macd_line > macd_signal,  # Momentum reversal
                        c > ema_fast_val,  # Price above fast EMA
                        profit_ratio < -1.5  # Emergency stop
                    ]
                    
                    if any(exit_conditions):
                        exit_price = max(h, trailing_stop)
                        reason = "Stop hit" if exit_conditions[0] else \
                                 "MACD reversal" if exit_conditions[1] else \
                                 "EMA break" if exit_conditions[2] else \
                                 "Time limit" if exit_conditions[3] else "Emergency stop"
                        
                        # Generate buy signal (1) to exit short position
                        df.at[i, 'signal'] = 1
                        decision = (f"EXIT SHORT @ {exit_price:.5f} | "
                                    f"Profit: {entry_price-exit_price:.5f} ({reason})")
                        df.at[i, 'decision'] = decision
                        df.at[i, 'stop_loss'] = trailing_stop
                        df.at[i, 'trailing_stop'] = trailing_stop
                        
                        # Reset trade state
                        active_trade = False
                        trade_direction = None
                    else:
                        # Maintain short position (no signal change)
                        df.at[i, 'signal'] = 0  # No new action needed
                        df.at[i, 'stop_loss'] = trailing_stop
                        df.at[i, 'trailing_stop'] = trailing_stop
                        df.at[i, 'decision'] = f"Holding SHORT | Current Profit: {current_profit:.5f}"
            else:
                # No active trade and no entry - set signal to 0
                df.at[i, 'signal'] = 0

        except Exception as e:
            error_msg = f"Error at index {i}: {str(e)}"
            print(error_msg)
            df.at[i, 'decision'] = error_msg
            df.at[i, 'signal'] = 0  # Ensure no signal on error
            continue

    # =====================
    # 4. Final Indicators Report
    # =====================
    if len(df) > 0:
        last = df.iloc[-1]
        print("\n=== STRATEGY METRICS ===")
        print(f"Trend Strength (ADX): {last['ADX']:.1f} {'(Strong)' if last['ADX'] > 20 else '(Moderate)'}")
        print(f"Volatility (ATR): {last['ATR']:.5f} | ATR Multiplier: {atr_multiplier}")
        print(f"EMA({ema_fast}/{ema_slow}): {last['ema_fast']:.12f} / {last['ema_slow']:.12f}")
        
        if 'MACD_line' in df.columns and 'MACD_signal' in df.columns:
            macd_status = "Bullish" if last['MACD_line'] > last['MACD_signal'] else "Bearish"
            print(f"MACD: {macd_status} ({last['MACD_line']:.12f} vs {last['MACD_signal']:.12f})")
        
        print(f"Volume Activity: {last['vroc']:.12f}x {'(High)' if last['vroc'] > 1.1 else '(Normal)'}")
        print(f"Last Decision: {decision}")
    
    return df, decision 

def place_order(signal, symbol, quantity, api_secret):
    global order_history
    request_path = '/api/v5/trade/order'
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    side = 'buy' if signal == 1 else 'sell'
    body = {
        "instId": symbol,
        "tdMode": "cash",
        "side": side,
        "ordType": "market",
        "sz": "145000" if side == 'sell' else "1.50000"
    }

    new_order = pd.DataFrame([{
        'timestamp': datetime.now(timezone.utc),
        'symbol': symbol,
        'side': side
    }])
    
    order_history = pd.concat(
        [order_history, new_order],
        ignore_index=True,
        copy=False
    ).astype({
        'timestamp': 'datetime64[ns, UTC]',
        'symbol': 'str',
        'side': 'str'
    })
    

    body_str = json.dumps(body, separators=(',', ':'))
    message = f"{timestamp}POST{request_path}{body_str}"
    signature = base64.b64encode(
        hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
    ).decode()
    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": API_PASSPHRASE,
        "x-simulated-trading": "0",  # Live trading
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(f'https://www.okx.com{request_path}', headers=headers, data=body_str)
        return response.json()
    except Exception as e:
        logging.error(f"Order failed: {str(e)}")
        return {'code': '50000', 'msg': 'Order processing error'}, order_history

def print_order_result(result):
    print("\n" + "=" * 40)
    print(f"{'ORDER EXECUTION':^40}")
    print("=" * 40)
    print(f"Raw Response: {result}")
    if result.get('code') == '0' and 'data' in result:
        print(f"Status:    Success (0)")
        print(f"Order ID:  {result['data'][0].get('ordId', 'N/A')}")
        print(f"Symbol:    {result['data'][0].get('instId', 'N/A')}")
        print(f"Size:      {result['data'][0].get('sz', 'N/A')}")
        print(f"Side:      {result['data'][0].get('side', 'N/A').upper()}")
    else:
        print(f"Error Code: {result.get('code', 'Unknown')}")
        print(f"Error Message: {result.get('msg', 'No error message')}")
        print(f"Request ID: {result.get('reqId', 'N/A')}")

async def execute_order(signal: int):
    """Async wrapper for order execution with error handling"""
    global df
    try:
        current_price = df['close'].iloc[-1]
        
        # Execute order
        result = await asyncio.to_thread(
            place_order,
            signal=signal,
            symbol=SYMBOL,
            quantity=1.5 if signal == 1 else 145000,
            api_secret=API_SECRET
        )
        
        print_order_result(result)
        logging.info(f"Order executed at price: {current_price}")
        
        return result
        
    except Exception as e:
        logging.error(f"Order execution failed: {str(e)}")
        return None
    
async def websocket_trading_bot():
    global df, last_processed_index, last_processed_minute
    global order_history

    df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=500)

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected DataFrame from fetch_ohlcv, but got: {type(df)} - {df}")
        raise ValueError("Invalid data fetched. Cannot proceed.")
    if df.empty:
        logging.error("Failed to fetch initial OHLCV data. Retrying...")
        await asyncio.sleep(5)
        return await websocket_trading_bot()
    ws_uri = "wss://ws.okx.com:8443/ws/v5/business"

    async with websockets.connect(ws_uri, ping_interval=25, ping_timeout=10, close_timeout=30) as websocket:
        subscribe_message = {
            "op": "subscribe",
            "args": [{"channel": "candle1m", "instId": SYMBOL}, 
                     {"channel": "trades", "instId": SYMBOL}]
                    
            
        }
        await websocket.send(json.dumps(subscribe_message))
        logging.info(f"Subscribed to {SYMBOL} 1m candles via websocket.")

        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
            except asyncio.TimeoutError:
                logging.warning("Timeout — sending ping manually")
                try:
                    await websocket.ping()
                except Exception as e:
                    logging.error(f"Ping failed: {e}")
                    break
            except websockets.ConnectionClosedError as e:
                logging.warning(f"WebSocket closed: code={e.code}, reason={e.reason}")
                break

            msg = json.loads(message)
             
            if 'arg' in msg and msg['arg']['channel'] == 'trades' and 'data' in msg:
               trade = msg['data'][0]
               global latest_price
               latest_price = float(trade['px'])


            if msg.get("event") == "ping":
                 pong = {"event": "pong"}
                 await websocket.send(json.dumps(pong))
                 logging.debug("Sent pong response to OKX ping")
                 continue  # Skip further processing for ping messages
            else:
                 logging.debug("Received non-candle websocket message: " + message)
            if 'data' in msg:
                candle_data = msg['data'][0]
                candle_ts = pd.to_datetime(int(candle_data[0]), unit='ms', utc=True)
                candle_minute = candle_ts.floor('min')

                if candle_minute != last_processed_minute:
                    last_processed_minute = candle_minute
                    new_candle = {
                        'timestamp': candle_ts,
                        'open': float(candle_data[1]),
                        'high': float(candle_data[2]),
                        'low': float(candle_data[3]),
                        'close': float(candle_data[4]),
                        'volume': float(candle_data[5])
                    }
                    logging.info(f"Received new candle: {new_candle}")

                    df, decision, signal = process_new_candle(new_candle)
                    logging.info(f"Decision: {decision} | Signal: {signal}")
                
                # --- Trading Decision ---
                if signal in [1, -1]:
                        last_order_side = order_history['side'].iloc[-1] if not order_history.empty else 'sell'
                        if (signal == 1 and last_order_side == 'sell') or (signal == -1 and last_order_side == 'buy'):
                            logging.info(f"Executing trade for signal {signal}")
                            task = asyncio.create_task(execute_order(signal))
                            try:
                                await asyncio.wait_for(task, timeout=10)
                            except asyncio.TimeoutError:
                                logging.error("Order execution timed out after 10 seconds")
                        else:
                            logging.info(f"Signal {signal} ignored due to current position: {last_order_side}")
            else:
                logging.debug("Received non-candle websocket message: " + message)
                logging.debug(f"Updated DataFrame:\n{df.tail(5)}")

async def main():
    while True:
        try:
            await websocket_trading_bot()
        except Exception as e:
            logging.critical(f"Critical failure: {str(e)}", exc_info=True)
            logging.info("Restarting in 30 seconds...")
            await asyncio.sleep(30)

if __name__ == '__main__':
    # Windows compatibility
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logging.info("Graceful shutdown initiated")
            break
        except Exception as e:
            logging.error(f"Catastrophic failure: {str(e)}")
            time.sleep(60)