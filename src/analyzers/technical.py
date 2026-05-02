"""
📊 Technical Indicators Analyzer
Hitung berbagai technical indicators dari OHLCV data menggunakan TA-Lib.
"""

import pandas as pd
import numpy as np
import talib
from typing import Optional


class TechnicalAnalyzer:
    """Hitung technical indicators dari OHLCV data."""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tambahkan SEMUA technical indicators ke DataFrame.
        
        Input DataFrame harus punya kolom: open, high, low, close, volume
        """
        df = df.copy()
        
        # Convert ke numpy arrays untuk TA-Lib
        open_p = df['open'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        # ====== TREND INDICATORS ======
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # ADX - trend strength (0-100, >25 = strong trend)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        
        # ====== MOMENTUM INDICATORS ======
        # RSI (0-100, <30=oversold, >70=overbought)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, 
                                         slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = signal
        df['MACD_hist'] = hist
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close,
                                    fastk_period=14, slowk_period=3,
                                    slowd_period=3)
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        
        # Williams %R (-100 to 0)
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # ====== VOLATILITY INDICATORS ======
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, 
                                              nbdevup=2, nbdevdn=2)
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower
        df['BB_width'] = (upper - lower) / middle
        df['BB_position'] = (close - lower) / (upper - lower)
        
        # ATR - Average True Range (volatility)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # ====== VOLUME INDICATORS ======
        df['OBV'] = talib.OBV(close, volume)
        df['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
        df['Volume_ratio'] = volume / df['Volume_SMA']
        
        # MFI (Money Flow Index) - 0-100, like RSI but with volume
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # ====== PRICE CHANGES ======
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_15'] = df['close'].pct_change(15)
        
        # ====== CANDLESTICK PATTERNS ======
        # Hasilnya: 0=tidak ada pattern, 100=bullish, -100=bearish
        df['DOJI'] = talib.CDLDOJI(open_p, high, low, close)
        df['HAMMER'] = talib.CDLHAMMER(open_p, high, low, close)
        df['ENGULFING'] = talib.CDLENGULFING(open_p, high, low, close)
        df['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(open_p, high, low, close)
        df['MORNING_STAR'] = talib.CDLMORNINGSTAR(open_p, high, low, close)
        df['EVENING_STAR'] = talib.CDLEVENINGSTAR(open_p, high, low, close)
        
        return df
    
    @staticmethod
    def get_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic trading signals berdasarkan technical indicators.
        Output: DataFrame dengan kolom signal (BUY/SELL/HOLD).
        """
        df = df.copy()
        
        # Initialize signal column
        df['signal'] = 'HOLD'
        df['signal_strength'] = 0  # 0-10 scale
        
        # ===== BUY SIGNALS =====
        buy_conditions = (
            (df['RSI'] < 35) &  # Oversold
            (df['MACD'] > df['MACD_signal']) &  # MACD bullish cross
            (df['close'] < df['BB_lower'] * 1.02)  # Near lower BB
        )
        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[buy_conditions, 'signal_strength'] = 7
        
        # Strong buy
        strong_buy = buy_conditions & (df['MFI'] < 30) & (df['ADX'] > 25)
        df.loc[strong_buy, 'signal_strength'] = 9
        
        # ===== SELL SIGNALS =====
        sell_conditions = (
            (df['RSI'] > 70) &  # Overbought
            (df['MACD'] < df['MACD_signal']) &  # MACD bearish cross
            (df['close'] > df['BB_upper'] * 0.98)  # Near upper BB
        )
        df.loc[sell_conditions, 'signal'] = 'SELL'
        df.loc[sell_conditions, 'signal_strength'] = 7
        
        # Strong sell
        strong_sell = sell_conditions & (df['MFI'] > 70) & (df['ADX'] > 25)
        df.loc[strong_sell, 'signal_strength'] = 9
        
        return df
    
    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict:
        """Get summary dari technical analysis."""
        latest = df.iloc[-1]
        
        # Trend analysis
        if latest['close'] > latest['SMA_50']:
            trend = "📈 UPTREND"
        else:
            trend = "📉 DOWNTREND"
        
        # RSI status
        rsi = latest['RSI']
        if rsi < 30:
            rsi_status = "🟢 OVERSOLD"
        elif rsi > 70:
            rsi_status = "🔴 OVERBOUGHT"
        else:
            rsi_status = "⚪ NEUTRAL"
        
        # MACD status
        if latest['MACD'] > latest['MACD_signal']:
            macd_status = "🟢 BULLISH"
        else:
            macd_status = "🔴 BEARISH"
        
        # Volatility (BB width relative to history)
        avg_bb_width = df['BB_width'].mean()
        if latest['BB_width'] > avg_bb_width * 1.5:
            volatility = "🔥 HIGH"
        elif latest['BB_width'] < avg_bb_width * 0.7:
            volatility = "❄️ LOW"
        else:
            volatility = "⚪ NORMAL"
        
        return {
            'price': latest['close'],
            'trend': trend,
            'rsi': rsi,
            'rsi_status': rsi_status,
            'macd_status': macd_status,
            'volatility': volatility,
            'adx': latest['ADX'],
            'signal': latest.get('signal', 'HOLD'),
            'signal_strength': latest.get('signal_strength', 0),
        }