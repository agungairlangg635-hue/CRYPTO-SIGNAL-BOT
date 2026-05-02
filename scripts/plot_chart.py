"""
Plot chart cantik dengan technical indicators.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np

from src.database.repository import PriceRepository
from src.analyzers.technical import TechnicalAnalyzer


def plot_full_chart(pair='btc_idr', last_n=200, save=True):
    """Plot comprehensive chart dengan semua indicators penting."""
    
    # Load data & hitung indicators
    df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=1000)
    
    if df.empty:
        print(f"❌ No data for {pair}")
        return
    
    df = TechnicalAnalyzer.add_all_indicators(df)
    df = TechnicalAnalyzer.get_signals(df)
    
    # Ambil N candles terakhir untuk display
    df = df.tail(last_n).reset_index(drop=True)
    
    # Setup figure dengan 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), 
                              gridspec_kw={'height_ratios': [3, 1, 1, 1]},
                              sharex=True)
    
    fig.suptitle(f'🇮🇩 {pair.upper().replace("_", "/")} Technical Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ============= CHART 1: Price + Bollinger + MA =============
    ax1 = axes[0]
    
    # Price line
    ax1.plot(df.index, df['close'], 'b-', linewidth=1.5, label='Close Price')
    
    # Moving averages
    ax1.plot(df.index, df['SMA_20'], 'orange', linewidth=1, alpha=0.8, label='SMA 20')
    ax1.plot(df.index, df['SMA_50'], 'red', linewidth=1, alpha=0.8, label='SMA 50')
    
    # Bollinger Bands
    ax1.fill_between(df.index, df['BB_lower'], df['BB_upper'], 
                      alpha=0.15, color='gray', label='Bollinger Bands')
    ax1.plot(df.index, df['BB_upper'], 'gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.plot(df.index, df['BB_lower'], 'gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # Plot BUY/SELL signals
    buy_signals = df[df['signal'] == 'BUY']
    sell_signals = df[df['signal'] == 'SELL']
    
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                    marker='^', s=100, color='green', zorder=5,
                    label=f'BUY ({len(buy_signals)})', edgecolors='darkgreen')
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                    marker='v', s=100, color='red', zorder=5,
                    label=f'SELL ({len(sell_signals)})', edgecolors='darkred')
    
    # Format y-axis ke Rupiah
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'Rp {x/1_000_000:.0f}jt' if x >= 1_000_000 else f'Rp {x:,.0f}')
    )
    
    ax1.set_ylabel('Harga (IDR)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Action + Bollinger Bands + Trading Signals', fontsize=11)
    
    # ============= CHART 2: Volume =============
    ax2 = axes[1]
    
    colors = ['green' if c >= o else 'red' 
              for c, o in zip(df['close'], df['open'])]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
    ax2.plot(df.index, df['Volume_SMA'], 'orange', linewidth=1.5, label='Volume SMA 20')
    
    ax2.set_ylabel('Volume', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Volume', fontsize=11)
    
    # ============= CHART 3: RSI =============
    ax3 = axes[2]
    
    ax3.plot(df.index, df['RSI'], 'purple', linewidth=1.5, label='RSI')
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax3.axhline(50, color='gray', linestyle='-', alpha=0.3)
    ax3.fill_between(df.index, 70, 100, alpha=0.1, color='red')
    ax3.fill_between(df.index, 0, 30, alpha=0.1, color='green')
    
    ax3.set_ylabel('RSI', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('RSI (Relative Strength Index)', fontsize=11)
    
    # ============= CHART 4: MACD =============
    ax4 = axes[3]
    
    ax4.plot(df.index, df['MACD'], 'blue', linewidth=1.5, label='MACD')
    ax4.plot(df.index, df['MACD_signal'], 'red', linewidth=1.5, label='Signal')
    
    # MACD histogram (bar chart)
    colors_hist = ['green' if h >= 0 else 'red' for h in df['MACD_hist']]
    ax4.bar(df.index, df['MACD_hist'], color=colors_hist, alpha=0.5, label='Histogram')
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    ax4.set_ylabel('MACD', fontsize=11)
    ax4.set_xlabel('Candles (terbaru di kanan)', fontsize=11)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('MACD (Moving Average Convergence Divergence)', fontsize=11)
    
    plt.tight_layout()
    
    # Save chart
    if save:
        output_dir = Path('docs/images')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pair}_technical_chart.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"✅ Chart saved to: {output_file}")
    
    plt.show()


def main():
    print("=" * 60)
    print("📈 Generating Technical Charts...")
    print("=" * 60)
    
    pairs = ['btc_idr', 'eth_idr', 'bnb_idr', 'sol_idr', 'xrp_idr']
    
    for pair in pairs:
        print(f"\n📊 Plotting {pair.upper()}...")
        try:
            plot_full_chart(pair, last_n=200, save=True)
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    print("\n🎉 All charts generated!")
    print("📁 Charts saved at: docs/images/")


if __name__ == "__main__":
    main()