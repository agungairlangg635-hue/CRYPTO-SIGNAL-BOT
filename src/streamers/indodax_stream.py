"""🇮🇩 Indodax Live Crypto Tracker."""

import requests
import time
import os
from datetime import datetime


def get_prices():
    """Ambil harga semua crypto dari Indodax."""
    response = requests.get("https://indodax.com/api/tickers", timeout=10)
    return response.json()['tickers']


def format_rp(amount):
    """Format Rupiah dengan separator titik."""
    return f"Rp {amount:,.0f}".replace(',', '.')


def main():
    pairs = {
        'btc_idr': ' Bitcoin',
        'eth_idr': ' Ethereum',
        'bnb_idr': ' BNB',
        'sol_idr': ' Solana',
        'xrp_idr': ' XRP',
        'doge_idr': ' Dogecoin',
    }
    
    prev_prices = {}
    
    print("🇮🇩 Memulai Indodax Tracker...")
    time.sleep(1)
    
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 75)
            print("🇮🇩  INDODAX LIVE TRACKER  🇮🇩".center(75))
            print(f" {datetime.now().strftime('%H:%M:%S WIB')}".center(75))
            print("=" * 75)
            print()
            
            tickers = get_prices()
            
            for pair, name in pairs.items():
                if pair not in tickers:
                    continue
                
                last = float(tickers[pair]['last'])
                
                # Trend dari update sebelumnya
                prev = prev_prices.get(pair, last)
                if last > prev:
                    arrow = "green"
                elif last < prev:
                    arrow = "red"
                else:
                    arrow = "white"
                
                print(f"  {name:<15} {format_rp(last):<22} {arrow}")
                prev_prices[pair] = last
            
            print()
            print("=" * 75)
            print(" Press Ctrl+C to stop | Refresh tiap 5 detik".center(75))
            print("=" * 75)
            
            time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n\n Bye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()