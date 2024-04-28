import ta
import pandas as pd

def clean_ds(df):
    df = df.copy()
    for i in range(1, 6):
        df[f'X_t-{i}'] = df['Close'].shift(i)

    # Shift Close Column up by 5 rows
    df['Pt_5'] = df['Close'].shift(-5)

    # Add RSI
    rsi_data = ta.momentum.RSIIndicator(close=df['Close'], window=28)
    df['RSI'] = rsi_data.rsi()

    # Y Labels
    df['Y_BUY'] = (df['Close'] < df['Pt_5'])
    df['Y_SELL'] = df['Close'] > df['Pt_5']

    return df