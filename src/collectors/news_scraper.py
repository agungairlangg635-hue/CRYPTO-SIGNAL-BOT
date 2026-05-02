"""
🇮🇩 News Scraper untuk Berita Crypto Indonesia (v2 - Reliable)
Pakai sumber yang dijamin punya data crypto.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import re


class CryptoNewsScraper:
    """Scraper berita crypto multi-source."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'id-ID,id;q=0.9,en;q=0.8',
        })
        
        # Crypto keywords (LEBIH KETAT)
        self.crypto_keywords = [
            # Major coins
            'bitcoin', 'btc', 'ethereum', 'eth ', 'ether ',
            'solana', 'sol ', 'cardano', 'ada ', 'binance',
            'dogecoin', 'doge', 'xrp', 'ripple', 'tether',
            
            # General crypto terms
            'crypto', 'kripto', 'cryptocurrency', 'altcoin',
            'blockchain', 'aset kripto', 'mata uang kripto',
            'mata uang digital', 'aset digital',
            
            # Indonesian exchanges & platforms
            'indodax', 'tokocrypto', 'pintu', 'reku',
            
            # Crypto specific
            'usdt', 'stablecoin', 'defi', 'web3', 'nft',
            'satoshi', 'mining', 'staking', 'wallet kripto'
        ]
    
    def is_crypto_related(self, *texts):
        """Cek apakah text berkaitan dengan crypto. Lebih strict."""
        # Combine all texts
        full_text = ' '.join([str(t) for t in texts if t]).lower()
        
        # Cek kata kunci
        return any(keyword in full_text for keyword in self.crypto_keywords)
    
    # ========== SOURCE 1: CoinDesk RSS ==========
    def scrape_coindesk(self):
        """CoinDesk - dijamin crypto, popular global."""
        print("📰 Scraping CoinDesk...")
        articles = []
        
        try:
            feed = feedparser.parse('https://www.coindesk.com/arc/outboundfeeds/rss/')
            
            for entry in feed.entries[:30]:
                articles.append({
                    'source': 'CoinDesk',
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'summary': BeautifulSoup(
                        entry.get('summary', ''), 'html.parser'
                    ).get_text()[:500],
                    'published_at': entry.get('published', ''),
                    'scraped_at': datetime.now(),
                    'language': 'en'
                })
            
            print(f"   ✅ Got {len(articles)} articles")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        
        return articles
    
    # ========== SOURCE 2: Cointelegraph RSS ==========
    def scrape_cointelegraph(self):
        """Cointelegraph - 100% crypto news."""
        print("📰 Scraping Cointelegraph...")
        articles = []
        
        try:
            feed = feedparser.parse('https://cointelegraph.com/rss')
            
            for entry in feed.entries[:30]:
                articles.append({
                    'source': 'Cointelegraph',
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'summary': BeautifulSoup(
                        entry.get('summary', ''), 'html.parser'
                    ).get_text()[:500],
                    'published_at': entry.get('published', ''),
                    'scraped_at': datetime.now(),
                    'language': 'en'
                })
            
            print(f"   ✅ Got {len(articles)} articles")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        
        return articles
    
    # ========== SOURCE 3: Decrypt.co RSS ==========
    def scrape_decrypt(self):
        """Decrypt - quality crypto journalism."""
        print("📰 Scraping Decrypt.co...")
        articles = []
        
        try:
            feed = feedparser.parse('https://decrypt.co/feed')
            
            for entry in feed.entries[:30]:
                articles.append({
                    'source': 'Decrypt',
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'summary': BeautifulSoup(
                        entry.get('summary', ''), 'html.parser'
                    ).get_text()[:500],
                    'published_at': entry.get('published', ''),
                    'scraped_at': datetime.now(),
                    'language': 'en'
                })
            
            print(f"   ✅ Got {len(articles)} articles")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        
        return articles
    
    # ========== SOURCE 4: Detik (Better Filter) ==========
    def scrape_detik_search(self):
        """Detik dengan search yang lebih spesifik."""
        print("📰 Scraping Detik (search 'bitcoin')...")
        articles = []
        
        try:
            # Search untuk "bitcoin" lebih akurat
            url = "https://www.detik.com/search/searchall?query=bitcoin&siteid=2"
            response = self.session.get(url, timeout=20)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Cari semua article
            for article in soup.find_all('article', limit=20):
                try:
                    title_tag = article.find(['h2', 'h3'])
                    link_tag = article.find('a', href=True)
                    
                    if not title_tag or not link_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    article_url = link_tag['href']
                    
                    if not article_url.startswith('http'):
                        article_url = 'https://detik.com' + article_url
                    
                    # Filter: HARUS ada keyword crypto di title
                    if self.is_crypto_related(title):
                        articles.append({
                            'source': 'Detik',
                            'title': title,
                            'url': article_url,
                            'summary': '',
                            'published_at': '',
                            'scraped_at': datetime.now(),
                            'language': 'id'
                        })
                except Exception:
                    continue
            
            print(f"   ✅ Got {len(articles)} crypto articles")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        
        return articles
    
    # ========== SOURCE 5: Kontan (Strict Filter) ==========
    def scrape_kontan_strict(self):
        """Kontan dengan filter STRICT — title harus mengandung crypto keyword."""
        print("📰 Scraping Kontan (strict filter)...")
        articles = []
        
        rss_urls = [
            'https://investasi.kontan.co.id/rss',
            'https://keuangan.kontan.co.id/rss',
        ]
        
        for rss_url in rss_urls:
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:50]:
                    title = entry.get('title', '')
                    summary_html = entry.get('summary', '')
                    summary = BeautifulSoup(summary_html, 'html.parser').get_text()
                    
                    # Filter STRICT: title atau summary HARUS mengandung crypto keyword
                    if self.is_crypto_related(title, summary):
                        articles.append({
                            'source': 'Kontan',
                            'title': title,
                            'url': entry.get('link', ''),
                            'summary': summary[:500],
                            'published_at': entry.get('published', ''),
                            'scraped_at': datetime.now(),
                            'language': 'id'
                        })
            except Exception as e:
                continue
        
        print(f"   ✅ Got {len(articles)} crypto articles")
        return articles
    
    # ========== Detect Coin Mentions ==========
    def detect_mentions(self, text):
        """Deteksi crypto coin yang disebut."""
        text_lower = (text or '').lower()
        
        # Pattern matching lebih akurat
        patterns = {
            'btc': r'\b(bitcoin|btc)\b',
            'eth': r'\b(ethereum|ether|eth)\b',
            'bnb': r'\b(bnb|binance coin)\b',
            'sol': r'\b(solana|sol)\b',
            'xrp': r'\b(ripple|xrp)\b',
            'doge': r'\b(dogecoin|doge)\b',
        }
        
        mentions = {}
        for coin, pattern in patterns.items():
            mentions[coin] = len(re.findall(pattern, text_lower))
        
        return mentions
    
    # ========== Main Scrape ==========
    def scrape_all(self):
        """Scrape dari semua sources."""
        print("\n" + "=" * 60)
        print("🇮🇩 SCRAPING CRYPTO NEWS - MULTI SOURCE")
        print("=" * 60 + "\n")
        
        all_articles = []
        
        # International (always reliable)
        all_articles.extend(self.scrape_coindesk())
        time.sleep(1)
        
        all_articles.extend(self.scrape_cointelegraph())
        time.sleep(1)
        
        all_articles.extend(self.scrape_decrypt())
        time.sleep(1)
        
        # Indonesia
        all_articles.extend(self.scrape_detik_search())
        time.sleep(1)
        
        all_articles.extend(self.scrape_kontan_strict())
        
        # To DataFrame
        df = pd.DataFrame(all_articles)
        
        if df.empty:
            return df
        
        # Detect mentions di title + summary
        df['full_text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
        df['mentions'] = df['full_text'].apply(self.detect_mentions)
        
        for coin in ['btc', 'eth', 'bnb', 'sol', 'xrp', 'doge']:
            df[f'mentions_{coin}'] = df['mentions'].apply(lambda x: x.get(coin, 0))
        
        df = df.drop(['mentions', 'full_text'], axis=1)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['url'], keep='first').reset_index(drop=True)
        
        return df


def main():
    scraper = CryptoNewsScraper()
    df = scraper.scrape_all()
    
    if df.empty:
        print("\n❌ No articles found")
        return
    
    # Save
    output_dir = Path('data/raw/news')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'crypto_news_{timestamp}.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SCRAPING SUMMARY")
    print("=" * 60)
    print(f"\n✅ Total articles: {len(df)}")
    print(f"💾 Saved to: {output_file}")
    
    # Per source
    print(f"\n📰 Articles per source:")
    for source, count in df['source'].value_counts().items():
        lang_indicator = "🇮🇩" if source in ['Detik', 'Kontan'] else "🌍"
        print(f"   {lang_indicator} {source:<20} {count} articles")
    
    # Per language
    print(f"\n🌐 Articles per language:")
    if 'language' in df.columns:
        for lang, count in df['language'].value_counts().items():
            flag = "🇮🇩" if lang == 'id' else "🌍"
            print(f"   {flag} {lang.upper()}: {count}")
    
    # Coin mentions
    print(f"\n🪙 Coin mentions (di title + summary):")
    for coin in ['btc', 'eth', 'bnb', 'sol', 'xrp', 'doge']:
        count = (df[f'mentions_{coin}'] > 0).sum()
        if count > 0:
            print(f"   {coin.upper():<5} {count} articles")
    
    # Top headlines (Indonesian first, then English)
    print(f"\n🇮🇩 Top Indonesian headlines:")
    id_articles = df[df['language'] == 'id'].head(5) if 'language' in df.columns else pd.DataFrame()
    if not id_articles.empty:
        for i, row in id_articles.iterrows():
            print(f"   • [{row['source']}] {row['title'][:80]}")
    else:
        print("   (No Indonesian articles found)")
    
    print(f"\n🌍 Top English headlines:")
    en_articles = df[df['language'] == 'en'].head(5) if 'language' in df.columns else df.head(5)
    for i, row in en_articles.iterrows():
        print(f"   • [{row['source']}] {row['title'][:80]}")
    
    print("\n" + "=" * 60)
    print("🎉 Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()