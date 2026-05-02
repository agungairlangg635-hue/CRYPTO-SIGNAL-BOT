"""SQLAlchemy models untuk database."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class PriceTick(Base):
    """Tick harga real-time."""
    __tablename__ = 'price_ticks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last = Column(Float, nullable=False)
    high_24h = Column(Float)
    low_24h = Column(Float)
    volume_24h = Column(Float)
    
    __table_args__ = (
        Index('idx_pair_timestamp', 'pair', 'timestamp'),
    )


class OHLCV(Base):
    """Candlestick OHLCV data."""
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False, index=True)
    resolution = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_pair_resolution_timestamp', 'pair', 'resolution', 'timestamp', unique=True),
    )


class TradingSignal(Base):
    """Trading signals dari ML model."""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    signal = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    price_at_signal = Column(Float, nullable=False)
    rsi = Column(Float)
    macd = Column(Float)
    sentiment_score = Column(Float)


class NewsArticle(Base):
    """News articles dengan sentiment."""
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    url = Column(String(500), unique=True)
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_label = Column(String(20))
    sentiment_score = Column(Float)
    mentions_btc = Column(Integer, default=0)
    mentions_eth = Column(Integer, default=0)