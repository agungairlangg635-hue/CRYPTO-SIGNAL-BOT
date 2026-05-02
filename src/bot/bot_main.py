"""
Crypto Signal Bot - Telegram Bot Main Entry.

Production-ready bot with:
- Real-time price tracking
- ML-powered trading signals
- Multi-user subscription system
- Inline keyboard navigation
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analyzers.technical import TechnicalAnalyzer
from src.database.repository import PriceRepository
from src.models.feature_engineering import FeatureEngineer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MODEL_PATH = Path('src/models/saved/xgboost_signal_model.pkl')

PAIRS_INFO = {
    'btc_idr': {'name': 'Bitcoin', 'symbol': 'BTC'},
    'eth_idr': {'name': 'Ethereum', 'symbol': 'ETH'},
    'bnb_idr': {'name': 'Binance Coin', 'symbol': 'BNB'},
    'sol_idr': {'name': 'Solana', 'symbol': 'SOL'},
    'xrp_idr': {'name': 'Ripple', 'symbol': 'XRP'},
}


# Global model artifact (loaded once)
MODEL_ARTIFACT = None


def load_model():
    """Load ML model artifact globally."""
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is None:
        MODEL_ARTIFACT = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded: trained on {MODEL_ARTIFACT['training_date']}")
    return MODEL_ARTIFACT


def format_idr(amount: float) -> str:
    """Format number as IDR currency."""
    if amount >= 1_000_000_000:
        return f"Rp {amount/1_000_000_000:.2f}M"
    elif amount >= 1_000_000:
        return f"Rp {amount/1_000_000:.2f}jt"
    elif amount >= 1_000:
        return f"Rp {amount/1_000:.0f}rb"
    return f"Rp {amount:,.0f}"


def get_signal_for_pair(pair: str) -> dict:
    """Generate ML signal untuk pair tertentu."""
    try:
        model_artifact = load_model()
        
        prices_df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=500)
        if prices_df.empty:
            return None
        
        prices_df = TechnicalAnalyzer.add_all_indicators(prices_df)
        
        engineer = FeatureEngineer(prediction_horizon=24, threshold_multiplier=1.5)
        features_df, _ = engineer.build_dataset(prices_df)
        
        if features_df.empty:
            return None
        
        # Get latest row only
        latest = features_df.iloc[-1:].copy()
        
        # Prepare features
        non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
        X = latest.drop(columns=[c for c in non_feature_cols if c in latest.columns])
        X = X.replace([float('inf'), float('-inf')], 0).fillna(0)
        
        expected_features = model_artifact['feature_names']
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        
        model = model_artifact['model']
        label_encoder = model_artifact['label_encoder']
        
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities.max()
        
        prediction_decoded = label_encoder.inverse_transform([prediction])[0]
        label_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        signal = label_map.get(prediction_decoded, 'HOLD')
        
        # Get current price
        current_price = prices_df['close'].iloc[-1]
        rsi = prices_df['RSI'].iloc[-1]
        macd = prices_df['MACD'].iloc[-1]
        
        return {
            'pair': pair,
            'signal': signal,
            'confidence': float(confidence),
            'price': float(current_price),
            'rsi': float(rsi) if not pd.isna(rsi) else None,
            'macd': float(macd) if not pd.isna(macd) else None,
        }
    
    except Exception as e:
        logger.error(f"Error generating signal for {pair}: {e}")
        return None


def get_signal_emoji(signal: str) -> str:
    """Get emoji for signal type."""
    return {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '⚪',
    }.get(signal, '⚪')


# ===== Command Handlers =====

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    
    welcome_msg = (
        f"👋 Halo {user.first_name}!\n\n"
        f"🇮🇩 *Crypto Signal Bot Indonesia*\n\n"
        f"Bot ini menggunakan Machine Learning untuk memberikan trading signals "
        f"crypto pasangan IDR di Indodax.\n\n"
        f"📊 *Coin yang di-track:*\n"
        f"BTC, ETH, BNB, SOL, XRP\n\n"
        f"*Pilih menu di bawah:*"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Semua Signals", callback_data="signals_all"),
            InlineKeyboardButton("💰 Cek Harga", callback_data="price_menu"),
        ],
        [
            InlineKeyboardButton("📈 Pilih Coin", callback_data="coin_menu"),
            InlineKeyboardButton("ℹ️ Tentang Bot", callback_data="about"),
        ],
    ]
    
    await update.message.reply_text(
        welcome_msg,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_msg = (
        "🤖 *Crypto Signal Bot - Commands*\n\n"
        "/start - Mulai & menu utama\n"
        "/signals - Lihat semua signal\n"
        "/signal <coin> - Signal untuk coin tertentu (contoh: /signal BTC)\n"
        "/price <coin> - Harga real-time (contoh: /price BTC)\n"
        "/about - Tentang project ini\n"
        "/help - Bantuan ini\n\n"
        "💡 *Cara pakai:*\n"
        "Klik tombol di /start untuk navigasi mudah, atau ketik command manual."
    )
    await update.message.reply_text(help_msg, parse_mode='Markdown')


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals command - show all signals."""
    await update.message.reply_text("⏳ Generating signals... tunggu sebentar")
    
    msg = "📊 *SIGNAL TERKINI - SEMUA COIN*\n\n"
    
    for pair, info in PAIRS_INFO.items():
        signal_data = get_signal_for_pair(pair)
        
        if signal_data is None:
            msg += f"⚠️ {info['name']}: Data tidak tersedia\n\n"
            continue
        
        emoji = get_signal_emoji(signal_data['signal'])
        confidence_pct = signal_data['confidence'] * 100
        
        msg += (
            f"{emoji} *{info['name']}* ({info['symbol']}/IDR)\n"
            f"   Signal: *{signal_data['signal']}*\n"
            f"   Confidence: {confidence_pct:.1f}%\n"
            f"   Harga: {format_idr(signal_data['price'])}\n"
        )
        
        if signal_data['rsi']:
            msg += f"   RSI: {signal_data['rsi']:.1f}\n"
        msg += "\n"
    
    msg += "_⚠️ Bukan saran finansial. DYOR._"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signal <coin> command."""
    if not context.args:
        await update.message.reply_text(
            "❓ Format salah. Contoh: `/signal BTC`",
            parse_mode='Markdown'
        )
        return
    
    coin = context.args[0].upper()
    pair = f"{coin.lower()}_idr"
    
    if pair not in PAIRS_INFO:
        available = ", ".join([info['symbol'] for info in PAIRS_INFO.values()])
        await update.message.reply_text(
            f"❌ Coin {coin} tidak tersedia.\n"
            f"Available: {available}"
        )
        return
    
    await update.message.reply_text(f"⏳ Generating signal untuk {coin}...")
    
    signal_data = get_signal_for_pair(pair)
    if signal_data is None:
        await update.message.reply_text(f"❌ Gagal generate signal untuk {coin}")
        return
    
    info = PAIRS_INFO[pair]
    emoji = get_signal_emoji(signal_data['signal'])
    confidence_pct = signal_data['confidence'] * 100
    
    msg = (
        f"{emoji} *{info['name']} Signal*\n\n"
        f"💰 Harga: {format_idr(signal_data['price'])}\n"
        f"🎯 Signal: *{signal_data['signal']}*\n"
        f"📊 Confidence: {confidence_pct:.1f}%\n\n"
        f"*Technical Indicators:*\n"
    )
    
    if signal_data['rsi']:
        rsi_status = "🔴 Overbought" if signal_data['rsi'] > 70 else "🟢 Oversold" if signal_data['rsi'] < 30 else "⚪ Neutral"
        msg += f"   RSI: {signal_data['rsi']:.1f} ({rsi_status})\n"
    
    if signal_data['macd']:
        msg += f"   MACD: {signal_data['macd']:.4f}\n"
    
    msg += "\n_⚠️ Bukan saran finansial. DYOR._"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /price <coin> command."""
    if not context.args:
        await update.message.reply_text(
            "❓ Format: `/price BTC`",
            parse_mode='Markdown'
        )
        return
    
    coin = context.args[0].upper()
    pair = f"{coin.lower()}_idr"
    
    if pair not in PAIRS_INFO:
        await update.message.reply_text(f"❌ Coin {coin} tidak tersedia")
        return
    
    df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=24)
    if df.empty:
        await update.message.reply_text(f"❌ Data harga tidak tersedia")
        return
    
    info = PAIRS_INFO[pair]
    current = df['close'].iloc[-1]
    yesterday = df['close'].iloc[0]
    change_pct = (current - yesterday) / yesterday * 100
    high_24h = df['high'].max()
    low_24h = df['low'].min()
    
    change_emoji = "📈" if change_pct >= 0 else "📉"
    
    msg = (
        f"💰 *{info['name']} ({info['symbol']}/IDR)*\n\n"
        f"💵 Harga: {format_idr(current)}\n"
        f"{change_emoji} 24h: {change_pct:+.2f}%\n"
        f"⬆️ High: {format_idr(high_24h)}\n"
        f"⬇️ Low: {format_idr(low_24h)}"
    )
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command."""
    msg = (
        "🤖 *Crypto Signal Bot Indonesia*\n\n"
        "Bot ini adalah portfolio project yang menggunakan:\n\n"
        "📊 *Tech Stack:*\n"
        "• Python 3.10\n"
        "• XGBoost ML model\n"
        "• 86 engineered features\n"
        "• 158 technical indicators (TA-Lib)\n"
        "• FinBERT + IndoBERT sentiment\n"
        "• CCXT untuk Indodax API\n"
        "• python-telegram-bot\n\n"
        "📈 *Data:*\n"
        "• 25,000+ historical candles\n"
        "• 5 pairs (BTC, ETH, BNB, SOL, XRP)\n"
        "• Real-time updates\n\n"
        "⚠️ *Disclaimer:*\n"
        "Bot ini untuk edukasi & demo portfolio. Bukan saran finansial. "
        "Trading crypto memiliki risiko tinggi."
    )
    await update.message.reply_text(msg, parse_mode='Markdown')


# ===== Callback Handlers =====

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "signals_all":
        await query.edit_message_text("⏳ Generating signals...")
        
        msg = "📊 *SIGNAL TERKINI - SEMUA COIN*\n\n"
        for pair, info in PAIRS_INFO.items():
            signal_data = get_signal_for_pair(pair)
            if signal_data:
                emoji = get_signal_emoji(signal_data['signal'])
                conf = signal_data['confidence'] * 100
                msg += (
                    f"{emoji} *{info['symbol']}*: {signal_data['signal']} "
                    f"({conf:.0f}%) - {format_idr(signal_data['price'])}\n"
                )
        
        msg += "\n_⚠️ Bukan saran finansial._"
        
        keyboard = [[InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")]]
        await query.edit_message_text(
            msg, parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "coin_menu":
        keyboard = []
        for pair, info in PAIRS_INFO.items():
            keyboard.append([
                InlineKeyboardButton(
                    f"📊 {info['name']} ({info['symbol']})",
                    callback_data=f"coin_{pair}"
                )
            ])
        keyboard.append([InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")])
        
        await query.edit_message_text(
            "📈 *Pilih Coin:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("coin_"):
        pair = data.replace("coin_", "")
        info = PAIRS_INFO.get(pair)
        if info:
            await query.edit_message_text(f"⏳ Loading {info['name']}...")
            
            signal_data = get_signal_for_pair(pair)
            if signal_data:
                emoji = get_signal_emoji(signal_data['signal'])
                conf = signal_data['confidence'] * 100
                
                msg = (
                    f"{emoji} *{info['name']} ({info['symbol']}/IDR)*\n\n"
                    f"💰 Harga: {format_idr(signal_data['price'])}\n"
                    f"🎯 Signal: *{signal_data['signal']}*\n"
                    f"📊 Confidence: {conf:.1f}%\n"
                )
                
                if signal_data['rsi']:
                    msg += f"📈 RSI: {signal_data['rsi']:.1f}\n"
                if signal_data['macd']:
                    msg += f"📊 MACD: {signal_data['macd']:.4f}\n"
                
                msg += "\n_⚠️ Bukan saran finansial._"
                
                keyboard = [
                    [InlineKeyboardButton("🔄 Refresh", callback_data=f"coin_{pair}")],
                    [InlineKeyboardButton("🔙 Pilih Coin Lain", callback_data="coin_menu")],
                ]
                await query.edit_message_text(
                    msg, parse_mode='Markdown',
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
    
    elif data == "price_menu":
        msg = "💰 *HARGA REAL-TIME*\n\n"
        for pair, info in PAIRS_INFO.items():
            df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=24)
            if not df.empty:
                current = df['close'].iloc[-1]
                yesterday = df['close'].iloc[0]
                change = (current - yesterday) / yesterday * 100
                emoji_change = "📈" if change >= 0 else "📉"
                msg += (
                    f"{emoji_change} *{info['symbol']}*: {format_idr(current)} "
                    f"({change:+.2f}%)\n"
                )
        
        keyboard = [[InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")]]
        await query.edit_message_text(
            msg, parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "about":
        msg = (
            "🤖 *Crypto Signal Bot Indonesia*\n\n"
            "📊 *Tech Stack:*\n"
            "Python | XGBoost | TA-Lib | CCXT | "
            "FinBERT | IndoBERT | Telegram Bot API\n\n"
            "📈 *Data:* 25,000+ candles, 5 IDR pairs\n"
            "🎯 *Features:* 86 engineered features\n"
            "🤖 *Model:* XGBoost classifier\n\n"
            "Created as portfolio project demonstrating "
            "end-to-end ML system development.\n\n"
            "⚠️ *Disclaimer:* Edukasi & demo only."
        )
        keyboard = [[InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")]]
        await query.edit_message_text(
            msg, parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "main_menu":
        keyboard = [
            [
                InlineKeyboardButton("📊 Semua Signals", callback_data="signals_all"),
                InlineKeyboardButton("💰 Cek Harga", callback_data="price_menu"),
            ],
            [
                InlineKeyboardButton("📈 Pilih Coin", callback_data="coin_menu"),
                InlineKeyboardButton("ℹ️ Tentang Bot", callback_data="about"),
            ],
        ]
        await query.edit_message_text(
            "🇮🇩 *Crypto Signal Bot*\n\nPilih menu:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


# ===== Main =====

def main():
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in .env file!")
        sys.exit(1)
    
    logger.info("Starting Crypto Signal Bot...")
    
    # Pre-load model
    load_model()
    
    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("signals", signals_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot ready! Polling for messages...")
    
    # Run bot
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()