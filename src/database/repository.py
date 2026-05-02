"""Data access layer - simpan & ambil data dari DB."""

import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError

from .connection import get_session
from .models import PriceTick, OHLCV, TradingSignal, NewsArticle


class PriceRepository:
    """Repository untuk price data."""
    
    @staticmethod
    def save_tick(pair, last, high_24h=None, low_24h=None, volume_24h=None):
        """Simpan satu tick harga real-time."""
        session = get_session()
        try:
            tick = PriceTick(
                pair=pair, last=last,
                high_24h=high_24h, low_24h=low_24h,
                volume_24h=volume_24h,
            )
            session.add(tick)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving tick: {e}")
        finally:
            session.close()
    
    @staticmethod
    def save_ohlcv_batch(pair, resolution, df):
        """Save banyak candles sekaligus dari DataFrame."""
        session = get_session()
        saved = 0
        try:
            for _, row in df.iterrows():
                try:
                    record = OHLCV(
                        pair=pair, resolution=resolution,
                        timestamp=row['timestamp'],
                        open=row['open'], high=row['high'],
                        low=row['low'], close=row['close'],
                        volume=row['volume'],
                    )
                    session.add(record)
                    session.commit()
                    saved += 1
                except IntegrityError:
                    # Duplicate (sudah ada), skip
                    session.rollback()
                    continue
            return saved
        except Exception as e:
            session.rollback()
            print(f"❌ Error: {e}")
            return saved
        finally:
            session.close()
    
    @staticmethod
    def get_ohlcv_df(pair, resolution='1h', limit=500):
        """Ambil OHLCV sebagai DataFrame."""
        session = get_session()
        try:
            query = (session.query(OHLCV)
                     .filter(OHLCV.pair == pair)
                     .filter(OHLCV.resolution == resolution)
                     .order_by(desc(OHLCV.timestamp))
                     .limit(limit))
            
            data = [{
                'timestamp': r.timestamp,
                'open': r.open, 'high': r.high,
                'low': r.low, 'close': r.close,
                'volume': r.volume,
            } for r in query.all()]
            
            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        finally:
            session.close()
    
    @staticmethod
    def count_records(pair=None, resolution=None):
        """Hitung total records di OHLCV table."""
        session = get_session()
        try:
            query = session.query(OHLCV)
            if pair:
                query = query.filter(OHLCV.pair == pair)
            if resolution:
                query = query.filter(OHLCV.resolution == resolution)
            return query.count()
        finally:
            session.close()