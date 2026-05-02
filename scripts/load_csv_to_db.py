"""
Load semua CSV historical data ke SQLite database.
Run setelah indodax_historical.py selesai.
"""

import sys
from pathlib import Path

# Add parent directory ke Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.database.connection import init_db
from src.database.repository import PriceRepository


def main():
    print("=" * 65)
    print(" Load CSV Data → SQLite Database")
    print("=" * 65)
    
    # Step 1: Initialize database
    init_db()
    print()
    
    # Step 2: Cari semua CSV di data/raw/prices/
    csv_dir = Path('data/raw/prices')
    
    if not csv_dir.exists():
        print(f" Folder {csv_dir} tidak ada!")
        print("   Run dulu: python indodax_historical.py")
        return
    
    csv_files = list(csv_dir.glob('*.csv'))
    
    if not csv_files:
        print(f" Tidak ada CSV files di {csv_dir}")
        print("   Run dulu: python indodax_historical.py")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Step 3: Load setiap CSV ke DB
    total_saved = 0
    for csv_file in csv_files:
        # Parse filename: btcidr_1h.csv → pair=btc_idr, resolution=1h
        name = csv_file.stem  # btcidr_1h
        parts = name.rsplit('_', 1)
        
        if len(parts) != 2:
            print(f" Skip {csv_file.name} (format salah)")
            continue
        
        pair_raw, resolution = parts
        # Convert btcidr → btc_idr
        pair = pair_raw[:-3] + '_' + pair_raw[-3:]
        
        print(f"Loading {csv_file.name} → pair={pair}, resolution={resolution}")
        
        try:
            df = pd.read_csv(csv_file, parse_dates=['timestamp'])
            
            if df.empty:
                print(f"   Empty CSV, skip")
                continue
            
            print(f"   {len(df)} rows from CSV")
            
            saved = PriceRepository.save_ohlcv_batch(pair, resolution, df)
            total_saved += saved
            
            print(f"    Saved {saved} new records (sisanya duplicate, di-skip)")
        
        except Exception as e:
            print(f"    Error: {e}")
        
        print()
    
    # Step 4: Summary
    print("=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    print(f"   Total records saved: {total_saved:,}")
    
    # Count per pair
    pairs = ['btc_idr', 'eth_idr', 'bnb_idr', 'sol_idr', 'xrp_idr']
    print(f"\n   Records per pair:")
    for pair in pairs:
        count_1h = PriceRepository.count_records(pair, '1h')
        count_1d = PriceRepository.count_records(pair, '1d')
        if count_1h > 0 or count_1d > 0:
            print(f"     {pair:<12} 1h: {count_1h:>5} | 1d: {count_1d:>5}")
    
    print("=" * 65)
    print("\n Done! Data siap untuk analysis & ML training!")
    print("\n Verify: Buka file data/crypto_bot.db dengan SQLite Browser")
    print("   Download: https://sqlitebrowser.org/")


if __name__ == "__main__":
    main()