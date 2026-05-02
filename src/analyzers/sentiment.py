"""
🎭 Multilingual Crypto Sentiment Analysis
Analisis sentiment artikel crypto dalam Bahasa Indonesia & English.
"""

import pandas as pd
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


class CryptoSentimentAnalyzer:
    """Sentiment analyzer untuk crypto news multibahasa."""
    
    def __init__(self):
        print("🤖 Loading sentiment models...")
        print("   (First time akan download ~500MB, tunggu sebentar...)")
        
        # FinBERT untuk English (financial domain)
        try:
            self.en_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            print("   ✅ FinBERT loaded (English)")
        except Exception as e:
            print(f"   ⚠️ FinBERT error: {e}")
            self.en_analyzer = None
        
        # IndoBERT untuk Indonesia
        try:
            self.id_analyzer = pipeline(
                "sentiment-analysis",
                model="indolem/indobert-base-uncased",
                tokenizer="indolem/indobert-base-uncased"
            )
            print("   ✅ IndoBERT loaded (Indonesian)")
        except Exception as e:
            # Fallback ke multilingual model
            print(f"   ⚠️ IndoBERT error, fallback to multilingual...")
            self.id_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            print("   ✅ Multilingual BERT loaded (fallback)")
    
    def analyze_text(self, text, language='en'):
        """Analyze sentiment dari satu text."""
        if not text or pd.isna(text):
            return {'label': 'neutral', 'score': 0.5, 'sentiment_value': 0}
        
        # Truncate text (BERT max 512 tokens)
        text = str(text)[:512]
        
        try:
            if language == 'id' and self.id_analyzer:
                result = self.id_analyzer(text)[0]
            elif self.en_analyzer:
                result = self.en_analyzer(text)[0]
            else:
                return {'label': 'neutral', 'score': 0.5, 'sentiment_value': 0}
            
            # Map label ke value (-1, 0, 1)
            label = result['label'].lower()
            
            if 'positive' in label or 'bullish' in label or label in ['4 stars', '5 stars']:
                sentiment_value = 1
            elif 'negative' in label or 'bearish' in label or label in ['1 star', '2 stars']:
                sentiment_value = -1
            else:
                sentiment_value = 0
            
            return {
                'label': result['label'],
                'score': result['score'],
                'sentiment_value': sentiment_value
            }
        
        except Exception as e:
            return {'label': 'neutral', 'score': 0.5, 'sentiment_value': 0}
    
    def analyze_dataframe(self, df, text_col='title'):
        """Analyze sentiment untuk semua artikel di DataFrame."""
        print(f"\n🎭 Analyzing sentiment for {len(df)} articles...")
        
        results = []
        for i, row in df.iterrows():
            language = row.get('language', 'en')
            text = row[text_col]
            
            result = self.analyze_text(text, language)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(df)}")
        
        # Add results to DataFrame
        df = df.copy()
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        df['sentiment_value'] = [r['sentiment_value'] for r in results]
        
        print(f"   ✅ Done!")
        return df


def main():
    from pathlib import Path
    
    # Load clean data
    input_file = Path('data/processed/news/crypto_news_clean.csv')
    
    if not input_file.exists():
        print(f"❌ {input_file} not found!")
        print("   Run dulu: python scripts/clean_news_data.py")
        return
    
    df = pd.read_csv(input_file)
    print(f"📂 Loaded {len(df)} articles\n")
    
    # Analyze
    analyzer = CryptoSentimentAnalyzer()
    df_with_sentiment = analyzer.analyze_dataframe(df, text_col='title')
    
    # Save
    output_file = Path('data/processed/news/crypto_news_with_sentiment.csv')
    df_with_sentiment.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SENTIMENT SUMMARY")
    print("=" * 60)
    
    sentiment_counts = df_with_sentiment['sentiment_value'].value_counts()
    
    bullish = sentiment_counts.get(1, 0)
    bearish = sentiment_counts.get(-1, 0)
    neutral = sentiment_counts.get(0, 0)
    total = len(df_with_sentiment)
    
    print(f"\n🟢 Bullish:  {bullish:>3} ({bullish/total*100:.1f}%)")
    print(f"🔴 Bearish:  {bearish:>3} ({bearish/total*100:.1f}%)")
    print(f"⚪ Neutral:  {neutral:>3} ({neutral/total*100:.1f}%)")
    
    avg_sentiment = df_with_sentiment['sentiment_value'].mean()
    print(f"\n📊 Average sentiment: {avg_sentiment:+.3f}")
    
    if avg_sentiment > 0.1:
        print("   🟢 Market mood: BULLISH")
    elif avg_sentiment < -0.1:
        print("   🔴 Market mood: BEARISH")
    else:
        print("   ⚪ Market mood: NEUTRAL")
    
    # Top bullish & bearish
    print(f"\n🟢 Top Bullish Headlines:")
    bullish_articles = df_with_sentiment[df_with_sentiment['sentiment_value'] == 1].head(3)
    for _, row in bullish_articles.iterrows():
        print(f"   • [{row['source']}] {row['title'][:75]}")
    
    print(f"\n🔴 Top Bearish Headlines:")
    bearish_articles = df_with_sentiment[df_with_sentiment['sentiment_value'] == -1].head(3)
    for _, row in bearish_articles.iterrows():
        print(f"   • [{row['source']}] {row['title'][:75]}")
    
    print("\n🎉 Sentiment analysis complete!")


if __name__ == "__main__":
    main()