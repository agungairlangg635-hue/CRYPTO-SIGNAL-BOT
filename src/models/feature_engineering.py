"""Feature Engineering Module dengan Adaptive Threshold."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Build training dataset dengan adaptive threshold (volatility-based)."""
    
    def __init__(
        self,
        prediction_horizon: int = 24,
        threshold_multiplier: float = 1.5,  # 1.5x ATR
    ) -> None:
        """
        Args:
            prediction_horizon: Berapa jam ke depan untuk prediksi
            threshold_multiplier: Multiplier untuk ATR-based threshold.
                Higher = strict (less BUY/SELL signals)
                Lower = lenient (more signals)
        """
        self.prediction_horizon = prediction_horizon
        self.threshold_multiplier = threshold_multiplier
    
    def add_lag_features(self, df, columns, lags=None):
        if lags is None:
            lags = [1, 3, 6, 12, 24]
        df = df.copy()
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    def add_rolling_features(self, df, columns, windows=None):
        if windows is None:
            windows = [6, 24, 168]
        df = df.copy()
        for col in columns:
            for window in windows:
                df[f'{col}_mean_{window}h'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}h'] = df[col].rolling(window).std()
        return df
    
    def add_temporal_features(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['session_europe'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['session_us'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        return df
    
    def add_price_position_features(self, df):
        df = df.copy()
        for window in [24, 168, 720]:
            rolling_high = df['high'].rolling(window).max()
            rolling_low = df['low'].rolling(window).min()
            df[f'price_position_{window}h'] = (
                (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            )
        if 'SMA_20' in df.columns:
            df['dist_sma_20_pct'] = (df['close'] - df['SMA_20']) / df['SMA_20'] * 100
        if 'SMA_50' in df.columns:
            df['dist_sma_50_pct'] = (df['close'] - df['SMA_50']) / df['SMA_50'] * 100
        return df
    
    def add_momentum_features(self, df):
        df = df.copy()
        for period in [1, 3, 6, 12, 24, 48]:
            df[f'return_{period}h'] = df['close'].pct_change(period)
        if 'ATR' in df.columns:
            df['vol_regime'] = df['ATR'].rolling(168).rank(pct=True)
            df['atr_pct'] = df['ATR'] / df['close']  # ATR sebagai % dari harga
        if 'volume' in df.columns:
            vol_ma_short = df['volume'].rolling(6).mean()
            vol_ma_long = df['volume'].rolling(24).mean()
            df['volume_momentum'] = (vol_ma_short - vol_ma_long) / (vol_ma_long + 1e-10)
        if 'RSI' in df.columns:
            df['rsi_short'] = df['RSI'].rolling(7).mean()
            df['rsi_long'] = df['RSI'].rolling(21).mean()
            df['rsi_divergence'] = df['rsi_short'] - df['rsi_long']
        return df
    
    def add_advanced_features(self, df):
        """Advanced features yang lebih predictive."""
        df = df.copy()
        
        # 1. Trend strength dengan multi-timeframe
        if all(c in df.columns for c in ['SMA_20', 'SMA_50']):
            df['ma_cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
            df['ma_distance'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        
        # 2. Bollinger squeeze (low volatility = breakout coming)
        if 'BB_width' in df.columns:
            df['bb_squeeze'] = df['BB_width'].rolling(20).rank(pct=True)
        
        # 3. RSI extreme zones
        if 'RSI' in df.columns:
            df['rsi_extreme_low'] = (df['RSI'] < 30).astype(int)
            df['rsi_extreme_high'] = (df['RSI'] > 70).astype(int)
        
        # 4. MACD signal cross
        if all(c in df.columns for c in ['MACD', 'MACD_signal']):
            df['macd_above_signal'] = (df['MACD'] > df['MACD_signal']).astype(int)
            df['macd_divergence'] = df['MACD'] - df['MACD_signal']
        
        # 5. Volume spike
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(24).mean()
            df['volume_spike'] = (df['volume'] > volume_ma * 1.5).astype(int)
        
        # 6. Higher highs / lower lows
        df['hh_24'] = (df['high'] > df['high'].rolling(24).max().shift(1)).astype(int)
        df['ll_24'] = (df['low'] < df['low'].rolling(24).min().shift(1)).astype(int)
        
        return df
    
    def add_sentiment_features(self, df, sentiment_df=None):
        df = df.copy()
        if sentiment_df is None or sentiment_df.empty:
            df['sentiment_score'] = 0.0
            df['sentiment_volume'] = 0
            df['bullish_ratio'] = 0.5
            return df
        if 'sentiment_value' not in sentiment_df.columns:
            df['sentiment_score'] = 0.0
            df['sentiment_volume'] = len(sentiment_df)
            df['bullish_ratio'] = 0.5
            return df
        avg_sentiment = sentiment_df['sentiment_value'].mean()
        n_articles = len(sentiment_df)
        bullish_count = (sentiment_df['sentiment_value'] == 1).sum()
        bullish_ratio = bullish_count / max(n_articles, 1)
        df['sentiment_score'] = avg_sentiment
        df['sentiment_volume'] = n_articles
        df['bullish_ratio'] = bullish_ratio
        return df
    
    def create_target_adaptive(self, df):
        """
        ADAPTIVE target — threshold based on rolling volatility (ATR).
        
        Lebih realistic karena:
        - Volatile crypto (SOL) butuh threshold lebih tinggi
        - Stable crypto (BTC) butuh threshold lebih rendah
        - Threshold beradaptasi seiring market regime
        """
        df = df.copy()
        
        df['future_close'] = df['close'].shift(-self.prediction_horizon)
        df['future_return'] = (df['future_close'] / df['close']) - 1
        
        # Adaptive threshold berdasarkan ATR sebagai % dari harga
        if 'ATR' in df.columns:
            atr_pct = df['ATR'] / df['close']
            # Smoothed ATR over last 24h untuk stability
            adaptive_threshold = atr_pct.rolling(24).mean() * self.threshold_multiplier
            
            # Fallback jika ATR not available
            adaptive_threshold = adaptive_threshold.fillna(0.02)
        else:
            # Default 2% kalau ATR tidak ada
            adaptive_threshold = pd.Series([0.02] * len(df), index=df.index)
        
        df['adaptive_threshold'] = adaptive_threshold
        
        # Apply target
        df['target'] = 0  # HOLD
        df.loc[df['future_return'] > adaptive_threshold, 'target'] = 1   # BUY
        df.loc[df['future_return'] < -adaptive_threshold, 'target'] = -1 # SELL
        
        return df
    
    def build_dataset(self, price_df, sentiment_df=None):
        df = price_df.copy()
        initial_rows = len(df)
        logger.info(f"Building features from {initial_rows} input rows")
        
        df = self.add_temporal_features(df)
        df = self.add_price_position_features(df)
        df = self.add_momentum_features(df)
        df = self.add_advanced_features(df)  # NEW: advanced features
        
        important_cols = [
            c for c in ['close', 'volume', 'RSI', 'MACD', 'BB_position']
            if c in df.columns
        ]
        df = self.add_lag_features(df, important_cols, lags=[1, 3, 6])
        
        if 'returns' in df.columns:
            df = self.add_rolling_features(df, ['returns'], windows=[6, 24])
        
        df = self.add_sentiment_features(df, sentiment_df)
        df = self.create_target_adaptive(df)  # NEW: adaptive target
        
        df = df.dropna(subset=['target'])
        
        # Drop helper columns
        drop_cols = [
            'timestamp', 'open', 'high', 'low', 'volume',
            'future_close', 'future_return', 'adaptive_threshold',
            'DOJI', 'HAMMER', 'ENGULFING', 'SHOOTING_STAR',
            'MORNING_STAR', 'EVENING_STAR',
        ]
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else None
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        df_features = df.drop(columns=cols_to_drop, errors='ignore')
        
        feature_count = df_features.shape[1] - 1
        logger.info(f"Generated {feature_count} features for {len(df_features)} samples")
        
        # Log target distribution
        target_dist = df_features['target'].value_counts(normalize=True).sort_index()
        for target, pct in target_dist.items():
            label = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}.get(target)
            logger.info(f"  {label}: {pct*100:.1f}%")
        
        return df_features, timestamps


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.database.repository import PriceRepository
    from src.analyzers.technical import TechnicalAnalyzer
    
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline (ADAPTIVE)")
    logger.info("=" * 60)
    
    sentiment_file = Path('data/processed/news/crypto_news_with_sentiment.csv')
    sentiment_df = None
    if sentiment_file.exists():
        sentiment_df = pd.read_csv(sentiment_file)
        logger.info(f"Loaded {len(sentiment_df)} articles with sentiment")
    
    engineer = FeatureEngineer(prediction_horizon=24, threshold_multiplier=1.5)
    
    output_dir = Path('data/processed/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = ['btc_idr', 'eth_idr', 'bnb_idr', 'sol_idr', 'xrp_idr']
    all_datasets = []
    
    for pair in pairs:
        logger.info(f"Processing {pair}")
        
        df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=10000)
        if df.empty:
            continue
        
        df = TechnicalAnalyzer.add_all_indicators(df)
        features_df, _ = engineer.build_dataset(df, sentiment_df)
        features_df['pair'] = pair
        
        output_file = output_dir / f'{pair}_training_data.csv'
        features_df.to_csv(output_file, index=False)
        all_datasets.append(features_df)
    
    if not all_datasets:
        return
    
    combined = pd.concat(all_datasets, ignore_index=True)
    combined_file = output_dir / 'all_pairs_training_data.csv'
    combined.to_csv(combined_file, index=False)
    
    logger.info(f"Combined dataset: {len(combined)} rows, {combined.shape[1]-2} features")


if __name__ == "__main__":
    main()