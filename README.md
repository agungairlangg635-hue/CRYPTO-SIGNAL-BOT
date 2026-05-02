# Crypto Signal Bot Indonesia

Sistem trading signal berbasis machine learning untuk pasar cryptocurrency Indonesia (Indodax). Project ini menggabungkan analisis teknikal, sentiment analysis multibahasa, dan classifier XGBoost untuk menghasilkan sinyal BUY/SELL/HOLD pada lima pasangan IDR utama: BTC, ETH, BNB, SOL, dan XRP.

Project ini dibangun sebagai latihan engineering end-to-end—mulai dari pengumpulan data, feature engineering, training model, sampai deployment berupa Telegram bot dan dashboard interaktif.

## Latar Belakang

Sebagian besar tutorial trading ML yang beredar online menampilkan akurasi yang menggiurkan—70%, 80%, bahkan 95%. Setelah saya implementasi sendiri, saya sadar angka-angka itu hampir selalu disebabkan data leakage atau metodologi validasi yang salah. Project ini sengaja dibangun dengan metodologi yang ketat untuk menghasilkan angka yang jujur, meskipun lebih rendah dari angka-angka tersebut.

Hasil akhirnya adalah sistem dengan akurasi 36% (di atas baseline random 33%) yang berhasil mengalahkan strategi buy-and-hold di 3 dari 5 pair pada periode out-of-sample. Bukan angka yang spektakuler, tapi angka yang bisa dipercaya.

## Hasil Backtest

Tabel di bawah membandingkan return strategi ML dengan buy-and-hold pada 1000 candle terakhir tiap pair (data yang tidak pernah dilihat saat training):

| Pair | Strategy | Buy-and-Hold | Selisih |
|------|----------|--------------|---------|
| BTC  | -3.41%   | +9.80%       | -13.21% |
| ETH  | -0.67%   | +6.62%       | -7.29%  |
| BNB  | -1.31%   | -3.46%       | **+2.15%** |
| SOL  | -2.67%   | -5.88%       | **+3.21%** |
| XRP  | -2.14%   | -4.36%       | **+2.22%** |

Pola yang muncul cukup menarik: strategi cenderung defensif. Saat market bullish kuat (BTC, ETH), strategi underperform karena terlalu sering exit dari posisi. Saat market bearish atau sideways (BNB, SOL, XRP), strategi berhasil menghindari drawdown besar dan mengalahkan buy-and-hold.

Maximum drawdown tetap di bawah 2% untuk semua pair, yang menunjukkan risk management berjalan baik meskipun akurasi prediksi modest.

## Komponen Sistem

**Data pipeline.** Data historis 1-jam diambil dari Indodax via CCXT dengan pagination. Total 27.920 candle dari 5 pair tersimpan di SQLite. News scraper mengumpulkan 79 artikel dari CoinDesk, Cointelegraph, Decrypt, Detik, dan Kontan.

**Feature engineering.** 86 fitur diturunkan dari data harga dan berita: indikator teknikal (RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, MFI), momentum multi-periode, cyclical encoding untuk fitur temporal, price position relative to rolling highs/lows, dan agregasi sentiment score.

**Sentiment analysis.** Berita berbahasa Inggris diproses dengan FinBERT (model finance-specific), berita Indonesia dengan IndoBERT. Hasil dikombinasikan menjadi sentiment score harian.

**Model.** XGBoost multi-class classifier (BUY/HOLD/SELL) dengan target adaptive berbasis ATR, bukan threshold tetap. Validasi dilakukan dengan per-pair temporal split—setiap pair displit 80/20 secara independen untuk mencegah training data dari satu pair bocor ke test set pair lain.

**Backtester.** Event-driven simulator dengan transaction cost 0.3% (sesuai taker fee Indodax), position sizing 20% dari modal, dan confidence threshold 0.40.

**Telegram bot.** Dibangun dengan python-telegram-bot. Mendukung command `/start`, `/signals`, `/signal <coin>`, `/price <coin>`, dan `/about`, plus inline keyboard untuk navigasi.

**Streamlit dashboard.** Interface ala TradingView dengan watchlist clickable, chart candlestick interaktif, signal panel real-time, order book simulation, dan trade history.

## Stack Teknologi

Python 3.10, XGBoost, TA-Lib, CCXT, SQLAlchemy, FinBERT, IndoBERT, HuggingFace Transformers, scikit-learn, python-telegram-bot, Streamlit, Plotly.

## Struktur Project
crypto-signal-bot/
├── src/
│   ├── analyzers/       # Technical & sentiment analysis
│   ├── bot/             # Telegram bot
│   ├── collectors/      # News scrapers
│   ├── database/        # SQLAlchemy models & repository
│   ├── models/          # ML pipeline & backtester
│   └── streamers/       # Historical & live data fetchers
├── scripts/             # Pipeline runner scripts
├── dashboard/           # Streamlit application
├── data/                # Raw & processed data
├── notebooks/           # EDA notebooks
└── docs/                # Charts & documentation

## Instalasi

---

## 🚀 Installation

### Prerequisites
- Python 3.10+, Anaconda, 4GB+ RAM

### Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/crypto-signal-bot.git
cd crypto-signal-bot

# Create environment
conda create -n cryptobot python=3.10
conda activate cryptobot
conda install -c conda-forge ta-lib

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env: add TELEGRAM_BOT_TOKEN dari @BotFather
```

### Run Pipeline

```bash
# Download data
python indodax_historical.py
python scripts/load_csv_to_db.py

# Train model
python src/models/feature_engineering.py
python src/models/train_signal_model.py
python scripts/run_backtest.py
```

### Run Applications

```bash
# Terminal 1 - Telegram Bot
python src/bot/bot_main.py

# Terminal 2 - Dashboard
streamlit run dashboard/app.py
```

Dashboard: `http://localhost:8501`

---

## 📚 Methodology

### 1. Data Collection
- 27,920 OHLCV candles dari 5 IDR pairs (1H)
- 79 news articles dari 5 sources (EN + ID)
- Real-time via CCXT

### 2. Feature Engineering (86 features)
- Technical indicators (RSI, MACD, BB, ATR, Stochastic, ADX)
- Momentum & volatility features
- Temporal features (cyclical encoding)
- Price position features
- Sentiment aggregation

### 3. Model Training
- **Algorithm:** XGBoost multi-class
- **Validation:** Per-pair temporal split (80/20)
- **Class balance:** Sample weights (sqrt scaling)

### 4. Backtesting
- Event-driven simulation
- 0.3% transaction cost (Indodax taker fee)
- 20% position sizing
- Confidence threshold 0.40

---

## 🎓 Lessons Learned

1. **Data Leakage Detection**
   - Initial validation: 100% win rate
   - Root cause: Combined dataset split
   - Fixed: Per-pair temporal validation

2. **Crypto Markets are Efficient**
   - Generic features = minimal edge
   - Need alternative data (order book, on-chain)

3. **Risk Management > Accuracy**
   - 36% accuracy + risk control = <2% drawdown
   - High confidence threshold filters noise

4. **Production Engineering Matters**
   - Logging, modularity, error handling
   - More valuable than 1-2% accuracy

5. **Honest Validation > Inflated Metrics**
   - Senior practitioners value methodology

---

## 📈 Future Improvements

- [ ] On-chain metrics integration
- [ ] LSTM/Transformer models
- [ ] Multi-timeframe signals
- [ ] Portfolio optimization
- [ ] Docker deployment
- [ ] Live paper trading

---

## ⚠️ Disclaimer

**For educational purposes only. NOT financial advice.**

Cryptocurrency trading involves substantial risk. DYOR (Do Your Own Research) before trading.

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 👤 Author

**Agung**
- LinkedIn: [your-linkedin](https://linkedin.com/in/yourusername)
- Email: your.email@example.com

---
