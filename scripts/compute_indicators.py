"""
Hitung technical indicators dari data di database, simpan hasilnya.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.database.repository import PriceRepository
from src.analyzers.technical import TechnicalAnalyzer


def main():
    print("=" * 65)
    print("📊 Compute Technical Indicators")
    print("=" * 65)
    
    pairs = ['btc_idr', 'eth_idr', 'bnb_idr', 'sol_idr', 'xrp_idr']
    
    # Folder untuk save hasil
    output_dir = Path('data/processed/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair in pairs:
        print(f"\n📈 Processing {pair.upper()}")
        print("-" * 50)
        
        # Load data dari DB
        df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=1000)
        
        if df.empty:
            print(f"  ⚠️ No data for {pair}")
            continue
        
        print(f"  📦 Loaded {len(df)} candles")
        
        # Hitung indicators
        df_with_indicators = TechnicalAnalyzer.add_all_indicators(df)
        
        # Generate signals
        df_with_signals = TechnicalAnalyzer.get_signals(df_with_indicators)
        
        # Save ke parquet (lebih efisien dari CSV)
        output_file = output_dir / f"{pair}_features.csv"
        df_with_signals.to_csv(output_file, index=False)
        print(f"  💾 Saved to: {output_file}")
        
        # Print summary
        summary = TechnicalAnalyzer.get_summary(df_with_signals)
        print(f"\n  📊 Summary:")
        print(f"     💰 Price:        Rp {summary['price']:,.0f}")
        print(f"     📈 Trend:        {summary['trend']}")
        print(f"     🎯 RSI:          {summary['rsi']:.1f} ({summary['rsi_status']})")
        print(f"     📊 MACD:         {summary['macd_status']}")
        print(f"     🌊 Volatility:   {summary['volatility']}")
        print(f"     💪 Trend Power:  ADX = {summary['adx']:.1f}")
        print(f"     🎯 Signal:       {summary['signal']} (strength: {summary['signal_strength']}/10)")
        
        # Count signals dalam 100 candles terakhir
        recent_signals = df_with_signals['signal'].iloc[-100:]
        buy_count = (recent_signals == 'BUY').sum()
        sell_count = (recent_signals == 'SELL').sum()
        print(f"     📈 Last 100 candles: {buy_count} BUY, {sell_count} SELL signals")
    
    print("\n" + "=" * 65)
    print("✅ Done! Indicators dihitung & disimpan ke data/processed/features/")
    print("=" * 65)


if __name__ == "__main__":
    main()