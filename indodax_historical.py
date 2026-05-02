"""
Indodax Historical Data Downloader (Paginated, 5000+ candles).
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class IndodaxHistorical:
    def __init__(self, output_dir='data/raw/prices'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.exchange = ccxt.indodax({'enableRateLimit': True})
        logger.info("Loading Indodax markets...")
        self.exchange.load_markets()
        logger.info(f"{len(self.exchange.markets)} markets loaded")
    
    def fetch_paginated(self, symbol, timeframe='1h', total_candles=5000):
        """Fetch banyak candles dengan multiple requests."""
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        end_ms = int(datetime.now().timestamp() * 1000)
        start_ms = end_ms - (total_candles * tf_ms)
        
        all_candles = []
        current_ms = start_ms
        request_count = 0
        max_requests = 20  # Safety limit
        
        while current_ms < end_ms and request_count < max_requests:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ms,
                    limit=1000,
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                request_count += 1
                
                last_ms = candles[-1][0]
                if last_ms <= current_ms:
                    break
                current_ms = last_ms + tf_ms
                
                logger.info(f"  Request #{request_count}: {len(all_candles):,} candles total")
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"  Error: {e}")
                time.sleep(3)
                continue
        
        if not all_candles:
            return None
        
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        
        return df
    
    def download(self, symbol, timeframe, target_candles):
        logger.info(f"Downloading {symbol} {timeframe} (target: {target_candles} candles)")
        
        df = self.fetch_paginated(symbol, timeframe, target_candles)
        
        if df is None or df.empty:
            logger.warning(f"  No data for {symbol}")
            return None
        
        pair = symbol.replace('/', '').lower()
        filename = self.output_dir / f"{pair}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        
        logger.info(
            f"  Got {len(df):,} candles | "
            f"Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to "
            f"{df['timestamp'].max().strftime('%Y-%m-%d')}"
        )
        logger.info(f"  Saved to: {filename}")
        return df


def main():
    logger.info("=" * 60)
    logger.info("INDODAX HISTORICAL DATA - EXTENDED DOWNLOAD")
    logger.info("=" * 60)
    
    fetcher = IndodaxHistorical()
    
    configs = [
        ('BTC/IDR', '1h', 5000),
        ('ETH/IDR', '1h', 5000),
        ('BNB/IDR', '1h', 5000),
        ('SOL/IDR', '1h', 5000),
        ('XRP/IDR', '1h', 5000),
        ('BTC/IDR', '1d', 730),
        ('ETH/IDR', '1d', 730),
        ('BNB/IDR', '1d', 730),
        ('SOL/IDR', '1d', 730),
    ]
    
    results = {}
    for symbol, timeframe, target in configs:
        df = fetcher.download(symbol, timeframe, target)
        if df is not None:
            key = f"{symbol.replace('/', '').lower()}_{timeframe}"
            results[key] = len(df)
        time.sleep(1)
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    total = sum(results.values())
    for key, count in results.items():
        logger.info(f"  {key:<20} {count:>6,} candles")
    logger.info(f"  TOTAL                {total:>6,} candles")


if __name__ == "__main__":
    main()