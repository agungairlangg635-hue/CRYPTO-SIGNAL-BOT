"""
🇮🇩 Indodax Real-time Crypto Price Tracker
Track harga crypto dalam Rupiah dari Indodax
"""

import requests
from datetime import datetime
import time
import os
from typing import Dict, Any


class IndodaxTracker:
    """Class untuk track harga crypto dari Indodax."""
    
    BASE_URL = "https://indodax.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoSignalBot/1.0 (Indonesia)'
        })
    
    def get_ticker(self, pair: str) -> Dict[str, Any]:
        """Ambil ticker untuk satu pair."""
        url = f"{self.BASE_URL}/ticker/{pair}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_all_tickers(self) -> Dict[str, Any]:
        """Ambil semua ticker sekaligus (lebih efisien!)."""
        url = f"{self.BASE_URL}/tickers"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_trades(self, pair: str) -> list:
        """Ambil recent trades."""
        url = f"{self.BASE_URL}/trades/{pair}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_orderbook(self, pair: str) -> Dict[str, Any]:
        """Ambil order book (buy/sell orders)."""
        url = f"{self.BASE_URL}/depth/{pair}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()


def format_rupiah(amount: float) -> str:
    """Format angka ke Rupiah dengan separator titik."""
    if amount >= 1_000_000_000:
        return f"Rp {amount/1_000_000_000:,.2f}M"
    elif amount >= 1_000_000:
        return f"Rp {amount/1_000_000:,.2f}jt"
    elif amount >= 1_000:
        return f"Rp {amount/1_000:,.2f}rb"
    else:
        return f"Rp {amount:,.0f}"


def format_rupiah_full(amount: float) -> str:
    """Format Rupiah lengkap dengan titik (Rp 1.500.000.000)."""
    return f"Rp {amount:,.0f}".replace(',', '.')


def calculate_change(last: float, low: float, high: float) -> float:
    """Hitung perubahan dari mid-price (perkiraan)."""
    mid = (low + high) / 2
    return ((last - mid) / mid) * 100 if mid > 0 else 0


def clear_screen():
    """Clear terminal untuk visualisasi yang lebih bersih."""
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    tracker = IndodaxTracker()
    
    # Top 10 crypto populer di Indodax
    pairs_to_track = [
        ('btc_idr', '₿ Bitcoin'),
        ('eth_idr', 'Ξ Ethereum'),
        ('bnb_idr', ' BNB'),
        ('sol_idr', ' Solana'),
        ('xrp_idr', ' XRP'),
        ('ada_idr', ' Cardano'),
        ('doge_idr', ' Dogecoin'),
        ('matic_idr', ' Polygon'),
        ('dot_idr', 'Polkadot'),
        ('avax_idr', ' Avalanche'),
    ]
    
    print("🇮🇩 Memulai Indodax Crypto Tracker...")
    time.sleep(1)
    
    try:
        iteration = 0
        previous_prices = {}
        
        while True:
            iteration += 1
            clear_screen()
            
            # Header
            print("=" * 95)
            print("🇮🇩  INDODAX REAL-TIME CRYPTO TRACKER  🇮🇩".center(95))
            print(f" {datetime.now().strftime('%A, %d %B %Y - %H:%M:%S WIB')}".center(95))
            print(f" Update #{iteration} | Auto-refresh setiap 10 detik | Press Ctrl+C to stop".center(95))
            print("=" * 95)
            
            # Table header
            print(f"\n{'#':<3} {'Asset':<15} {'Harga':<22} {'High 24h':<22} {'Low 24h':<22} {'Trend':<8}")
            print("-" * 95)
            
            try:
                # Ambil semua ticker sekaligus (1 request, lebih efisien!)
                all_data = tracker.get_all_tickers()
                tickers = all_data.get('tickers', {})
                
                for idx, (pair, name) in enumerate(pairs_to_track, 1):
                    try:
                        ticker = tickers.get(pair, {})
                        
                        if not ticker:
                            print(f"{idx:<3} {name:<15}  Data tidak tersedia")
                            continue
                        
                        last = float(ticker.get('last', 0))
                        high = float(ticker.get('high', 0))
                        low = float(ticker.get('low', 0))
                        
                        # Tentukan trend dibanding update sebelumnya
                        prev = previous_prices.get(pair, last)
                        if last > prev:
                            trend = "green"
                            color = "\033[92m"  # Green
                        elif last < prev:
                            trend = "red"
                            color = "\033[91m"  # Red
                        else:
                            trend = "white"
                            color = "\033[97m"  # White
                        
                        reset = "\033[0m"
                        
                        # Hitung % change perkiraan
                        change_pct = calculate_change(last, low, high)
                        
                        print(f"{idx:<3} {name:<15} "
                              f"{color}{format_rupiah_full(last):<22}{reset} "
                              f"{format_rupiah_full(high):<22} "
                              f"{format_rupiah_full(low):<22} "
                              f"{trend}")
                        
                        previous_prices[pair] = last
                        
                    except Exception as e:
                        print(f"{idx:<3} {name:<15}  Error: {e}")
                
                print("-" * 95)
                print(f"\n Tip: Format harga dalam Rupiah (Rp). Trend dibandingkan update sebelumnya.")
                print(f" Data source: Indodax.com (Exchange resmi Bappebti)")
                
            except requests.exceptions.RequestException as e:
                print(f"\n Network error: {e}")
                print("   Mencoba lagi dalam 10 detik...")
            
            # Wait 10 seconds
            time.sleep(10)
            
    except KeyboardInterrupt:
        clear_screen()
        print("\n" + "=" * 50)
        print(" Tracker dihentikan")
        print(f" Total updates: {iteration}")
        print(" Stopped cleanly!")
        print("=" * 50)


if __name__ == "__main__":
    main()