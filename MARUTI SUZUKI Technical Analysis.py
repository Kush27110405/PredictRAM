import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------- Fetch data ----------
ticker = "MARUTI.NS"
stock = yf.Ticker(ticker)

# Use explicit period/interval; auto_adjust=False to keep raw Close (set True if you want adjusted)
hist = stock.history(period="365d", interval="1d", auto_adjust=False)

# ---------- Indicator functions ----------
def calculate_rsi_wilder(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (RMA) using ewm with adjust=False and alpha=1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(prices):
    return {
        'SMA_20': prices.rolling(window=20).mean(),
        'SMA_50': prices.rolling(window=50).mean(),
        # use adjust=False to match typical EMA implementations (TradingView)
        'EMA_12': prices.ewm(span=12, adjust=False).mean(),
        'EMA_26': prices.ewm(span=26, adjust=False).mean()
    }

# ---------- Calculate indicators ----------
hist['RSI'] = calculate_rsi_wilder(hist['Close'], period=14)
ma_data = calculate_moving_averages(hist['Close'])
for k, v in ma_data.items():
    hist[k] = v

hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

# Drop rows with NaNs if needed
hist = hist.dropna()

# ---------- Plot (same as your layout) ----------
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
axes[0].plot(hist.index, hist['Close'], label='Close Price', linewidth=2)
axes[0].plot(hist.index, hist['SMA_20'], label='SMA 20', alpha=0.7)
axes[0].plot(hist.index, hist['SMA_50'], label='SMA 50', alpha=0.7)
axes[0].set_title(f'{ticker} - Price and Moving Averages')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist.index, hist['RSI'], linewidth=2)
axes[1].axhline(70, linestyle='--'); axes[1].axhline(30, linestyle='--')
axes[1].set_title('RSI (Wilder)'); axes[1].grid(True, alpha=0.3)

axes[2].plot(hist.index, hist['MACD'], label='MACD', linewidth=2)
axes[2].plot(hist.index, hist['MACD_Signal'], label='Signal', linewidth=2)
axes[2].bar(hist.index, hist['MACD'] - hist['MACD_Signal'], alpha=0.3)
axes[2].set_title('MACD'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------- Current values ----------
current_price = hist['Close'].iloc[-1]         # last daily close from Yahoo
sma_20 = hist['SMA_20'].iloc[-1]
current_rsi = hist['RSI'].iloc[-1]

print(f"Last daily Close: ₹{current_price:.2f}")
print(f"20-day SMA: ₹{sma_20:.2f}")
print(f"RSI (Wilder): {current_rsi:.2f}")

# ---------- (Optional) get intraday/latest price to match TradingView live tick ----------
# Use 1m intraday (if available) to get the most recent tick price:
try:
    intraday = stock.history(period='1d', interval='1m', auto_adjust=False)
    if not intraday.empty:
        latest_tick = intraday['Close'].iloc[-1]
        print(f"Latest intraday tick: ₹{latest_tick:.2f}")
except Exception:
    pass
