"""
🧹 Clean & Filter News Data
Filter artikel yang TIDAK related crypto, simpan yang clean.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import re


def is_real_crypto_article(title, summary=''):
    """
    Filter STRICT untuk crypto article.
    Harus ada keyword crypto + kontext yang masuk akal.
    """
    text = (str(title) + ' ' + str(summary)).lower()
    
    # ❌ EXCLUDE: keyword yang sering bikin false positive
    exclude_patterns = [
        r'\b(saham|stock)\b.*\b(ihsg|wall street)\b',  # Saham news
        r'\b(laba bersih|laba kuartal)\b',  # Earnings
        r'\b(rebound|naik turun)\b.*\b(ihsg)\b',  # Stock market
        r'tewas|bunuh|mayat|kriminal',  # Crime/sensational
        r'gosip|selebriti|model',  # Gossip
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, text):
            return False
    
    # ✅ INCLUDE: harus ada keyword crypto yang strong
    strong_crypto_keywords = [
        # Major coins (with word boundary)
        r'\b(bitcoin|btc)\b',
        r'\b(ethereum|eth)\b',
        r'\b(solana|sol)\b',
        r'\b(cardano|ada)\b',
        r'\b(binance coin|bnb)\b',
        r'\b(dogecoin|doge)\b',
        r'\b(ripple|xrp)\b',
        
        # Crypto terms
        r'\b(crypto|kripto)\b',
        r'\b(cryptocurrency)\b',
        r'\b(altcoin|altcoins)\b',
        r'\b(blockchain)\b',
        r'\b(stablecoin|tether|usdt)\b',
        r'\b(defi|nft)\b',
        r'\baset\s+kripto\b',
        r'\bmata\s+uang\s+(digital|kripto)\b',
        
        # Indonesian platforms
        r'\b(indodax|tokocrypto|pintu|reku)\b',
    ]
    
    for pattern in strong_crypto_keywords:
        if re.search(pattern, text):
            return True
    
    return False


def main():
    # Find latest news CSV
    news_dir = Path('data/raw/news')
    csv_files = sorted(news_dir.glob('crypto_news_*.csv'))
    
    if not csv_files:
        print("❌ No news CSV found! Run news_scraper.py first.")
        return
    
    latest_file = csv_files[-1]
    print(f"📂 Processing: {latest_file.name}\n")
    
    # Load
    df = pd.read_csv(latest_file)
    original_count = len(df)
    
    print(f"📊 Original articles: {original_count}")
    
    # Filter
    print("\n🧹 Filtering articles...")
    df['is_crypto'] = df.apply(
        lambda row: is_real_crypto_article(row['title'], row.get('summary', '')),
        axis=1
    )
    
    df_clean = df[df['is_crypto']].copy()
    df_excluded = df[~df['is_crypto']].copy()
    
    # Drop helper column
    df_clean = df_clean.drop('is_crypto', axis=1)
    
    print(f"\n✅ Clean articles: {len(df_clean)}")
    print(f"❌ Excluded:       {len(df_excluded)}")
    
    # Show what was excluded
    if not df_excluded.empty:
        print("\n🗑️  Excluded examples:")
        for _, row in df_excluded.head(5).iterrows():
            print(f"   • [{row['source']}] {row['title'][:80]}")
    
    # Save cleaned data
    output_dir = Path('data/processed/news')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'crypto_news_clean.csv'
    df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Saved clean data to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CLEAN DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📰 Per source:")
    for source, count in df_clean['source'].value_counts().items():
        flag = "🇮🇩" if source in ['Detik', 'Kontan'] else "🌍"
        print(f"   {flag} {source:<20} {count} articles")
    
    print(f"\n🌐 Per language:")
    if 'language' in df_clean.columns:
        for lang, count in df_clean['language'].value_counts().items():
            flag = "🇮🇩" if lang == 'id' else "🌍"
            print(f"   {flag} {lang.upper()}: {count}")
    
    print(f"\n🪙 Coin mentions:")
    for coin in ['btc', 'eth', 'bnb', 'sol', 'xrp', 'doge']:
        col = f'mentions_{coin}'
        if col in df_clean.columns:
            count = (df_clean[col] > 0).sum()
            if count > 0:
                print(f"   {coin.upper():<5} {count} articles")
    
    print(f"\n🎯 Sample clean articles:")
    for i, row in df_clean.head(8).iterrows():
        flag = "🇮🇩" if row.get('language', 'en') == 'id' else "🌍"
        print(f"   {flag} [{row['source']}] {row['title'][:80]}")
    
    print("\n" + "=" * 60)
    print("✅ Done! Data ready for sentiment analysis!")


if __name__ == "__main__":
    main()