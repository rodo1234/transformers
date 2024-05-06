import ta
import pandas as pd

def clean_ds(df):
    df = df.copy()
    for i in range(1, 6):
        df[f'X_t-{i}'] = df['Close'].shift(i)

    # Shift Close Column up by 5 rows
    df['Pt_5'] = df['Close'].shift(-5)

    # Adding RSI and EMA indicators with varied windows
    rsi_windows = [5, 10, 15, 28, 32]
    ema_windows = [5, 10, 15, 28, 32]
    for window in rsi_windows:
        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=window).rsi()
        df[f'RSI_{window}'] = rsi
    for window in ema_windows:
        ema = ta.trend.EMAIndicator(close=df['Close'], window=window).ema_indicator()
        df[f'EMA_{window}'] = ema
    
    # MACD with different settings
    macd_settings = [(12, 26, 9), (5, 35, 5), (7, 14, 3)]
    for fast, slow, signal in macd_settings:
        macd = ta.trend.MACD(close=df['Close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df[f'MACD_line_{fast}_{slow}_{signal}'] = macd.macd()
        df[f'MACD_signal_{fast}_{slow}_{signal}'] = macd.macd_signal()
        df[f'MACD_diff_{fast}_{slow}_{signal}'] = macd.macd_diff()
    
    # Bollinger Bands with different windows
    bollinger_settings = [(20, 2), (10, 1.5), (30, 3)]
    for window, dev in bollinger_settings:
        bollinger = ta.volatility.BollingerBands(close=df['Close'], window=window, window_dev=dev)
        df[f'Bollinger_mavg_{window}'] = bollinger.bollinger_mavg()
        df[f'Bollinger_hband_{window}'] = bollinger.bollinger_hband()
        df[f'Bollinger_lband_{window}'] = bollinger.bollinger_lband()
    
    # Stochastic Oscillator with different settings
    stoch_settings = [(14, 3), (10, 3), (20, 5)]
    for window, smooth in stoch_settings:
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=window, smooth_window=smooth)
        df[f'Stoch_%K_{window}'] = stoch.stoch()
        df[f'Stoch_%D_{window}'] = stoch.stoch_signal()
    
    # Average True Range with different windows
    atr_windows = [14, 10, 20]
    for window in atr_windows:
        atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'ATR_{window}'] = atr.average_true_range()
    
    # Commodity Channel Index with different windows
    cci_windows = [10, 20, 40]
    for window in cci_windows:
        cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'CCI_{window}'] = cci.cci()

    # Parabolic SAR with different step and max_step
    sar_steps = [(0.02, 0.2), (0.01, 0.1), (0.03, 0.3)]
    for step, max_step in sar_steps:
        sar = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=step, max_step=max_step)
        df[f'SAR_{step}_{max_step}'] = sar.psar()

    # Ichimoku Cloud with different settings
    ichimoku_settings = [(9, 26, 52), (7, 22, 44), (12, 30, 60)]
    for window1, window2, window3 in ichimoku_settings:
        ichimoku = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=window1, window2=window2, window3=window3)
        df[f'Ichimoku_A_{window1}_{window2}_{window3}'] = ichimoku.ichimoku_a()
        df[f'Ichimoku_B_{window1}_{window2}_{window3}'] = ichimoku.ichimoku_b()

    # # Pivot Points, Standard, with different windows
    # pivot_windows = [1, 3, 5]
    # for window in pivot_windows:
    #     pivot = ta.trend.PivotPointIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
    #     df[f'Pivot_Point_{window}'] = pivot.pivot_point()
    #     df[f'Resistance_1_{window}'] = pivot.resistance_1()
    #     df[f'Support_1_{window}'] = pivot.support_1()

    # Money Flow Index with different windows
    mfi_windows = [10, 14, 20]
    for window in mfi_windows:
        mfi = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=window)
        df[f'MFI_{window}'] = mfi.money_flow_index()

    # Chaikin Money Flow with different windows
    cmf_windows = [10, 20, 30]
    for window in cmf_windows:
        cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=window)
        df[f'CMF_{window}'] = cmf.chaikin_money_flow()
    
    # Williams %R with different windows
    williams_windows = [7, 14, 28]
    for window in williams_windows:
        williams = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=window)
        df[f'Williams_{window}'] = williams.williams_r()

    # Ultimate Oscillator with different settings
    ultimate_settings = [(7, 14, 28), (5, 10, 20), (10, 20, 30)]
    for short, medium, long in ultimate_settings:
        ultimate = ta.momentum.UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], 
                                                  window1=short, window2=medium, window3=long, weight1=4, weight2=2, weight3=1)
        df[f'Ultimate_Osc_{short}_{medium}_{long}'] = ultimate.ultimate_oscillator()

    # TRIX with different windows
    trix_windows = [15, 30, 45]
    for window in trix_windows:
        trix = ta.trend.TRIXIndicator(close=df['Close'], window=window)
        df[f'TRIX_{window}'] = trix.trix()

    # Keltner Channels with different windows
    keltner_windows = [(20, 2), (10, 1), (40, 2.5)]
    for window, atr_size in keltner_windows:
        keltner = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'],
                                               window=window, window_atr=window, original_version=False,
                                               fillna=False)
        df[f'Keltner_Center_{window}'] = keltner.keltner_channel_mband()
        df[f'Keltner_High_{window}'] = keltner.keltner_channel_hband()
        df[f'Keltner_Low_{window}'] = keltner.keltner_channel_lband()
        
    # Rate of Change (ROC) with different windows
    roc_windows = [10, 20, 30]
    for window in roc_windows:
        roc = ta.momentum.ROCIndicator(close=df['Close'], window=window)
        df[f'ROC_{window}'] = roc.roc()

    # Y Labels
    df['Y_BUY'] = (df['Close'] < df['Pt_5'])
    df['Y_SELL'] = df['Close'] > df['Pt_5']

    return df