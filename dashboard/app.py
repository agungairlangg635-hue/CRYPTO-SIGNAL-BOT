"""
Crypto Signal Bot - Professional Trading Dashboard.
TradingView-inspired interactive interface dengan full interactivity.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.technical import TechnicalAnalyzer
from src.database.repository import PriceRepository
from src.models.feature_engineering import FeatureEngineer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Crypto Signal Bot - Live Trading",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Custom CSS - Trading Pro Theme
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: #0a0e1a;
        color: #e2e8f0;
    }
    
    /* Header bar */
    .header-bar {
        background: linear-gradient(90deg, #1a1f2e 0%, #0f1419 100%);
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .live-indicator {
        display: inline-block;
        background: #22c55e;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Watchlist styling */
    .watchlist-item {
        background: #1a1f2e;
        padding: 12px 15px;
        margin: 6px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }
    
    .watchlist-item:hover {
        background: #242a3d;
        border-left: 3px solid #667eea;
        transform: translateX(3px);
    }
    
    .watchlist-active {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.2) 0%, transparent 100%);
        border-left: 3px solid #667eea !important;
    }
    
    /* Trading panel */
    .trade-panel {
        background: #1a1f2e;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        color: white;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        color: white;
    }
    
    /* Buy/Sell action buttons */
    div[data-testid="column"] .stButton > button {
        width: 100%;
        font-weight: 700;
        height: 50px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s;
    }
    
    /* Order book styling */
    .order-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 12px;
        font-family: 'Consolas', monospace;
        font-size: 12px;
    }
    
    .ask-row {
        background: rgba(239, 68, 68, 0.05);
        border-left: 2px solid rgba(239, 68, 68, 0.3);
    }
    
    .bid-row {
        background: rgba(34, 197, 94, 0.05);
        border-left: 2px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Indicator bars */
    .indicator-bar {
        background: #2a3142;
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .indicator-fill {
        height: 100%;
        background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%);
        transition: width 0.3s ease;
    }
    
    /* Stat cards */
    .stat-card {
        background: #1a1f2e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .stat-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .stat-value {
        font-size: 22px;
        font-weight: 700;
    }
    
    .stat-positive { color: #22c55e; }
    .stat-negative { color: #ef4444; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Trade history table */
    .trade-row {
        display: grid;
        grid-template-columns: 80px 60px 1fr 1fr;
        padding: 8px 12px;
        font-size: 12px;
        font-family: 'Consolas', monospace;
        border-bottom: 1px solid #2a3142;
    }
    
    .trade-buy { color: #22c55e; }
    .trade-sell { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Configuration
# ============================================================

PAIRS_INFO = {
    'btc_idr': {'name': 'Bitcoin', 'symbol': 'BTC', 'color': '#F7931A', 'icon': '₿'},
    'eth_idr': {'name': 'Ethereum', 'symbol': 'ETH', 'color': '#627EEA', 'icon': 'Ξ'},
    'bnb_idr': {'name': 'Binance Coin', 'symbol': 'BNB', 'color': '#F3BA2F', 'icon': 'B'},
    'sol_idr': {'name': 'Solana', 'symbol': 'SOL', 'color': '#14F195', 'icon': 'S'},
    'xrp_idr': {'name': 'Ripple', 'symbol': 'XRP', 'color': '#23292F', 'icon': 'X'},
}

MODEL_PATH = Path('src/models/saved/xgboost_signal_model.pkl')


# ============================================================
# Session State Initialization
# ============================================================

if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = 'btc_idr'
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = '1H'
if 'show_modal' not in st.session_state:
    st.session_state.show_modal = False
if 'modal_action' not in st.session_state:
    st.session_state.modal_action = None
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []


# ============================================================
# Data Functions (Cached)
# ============================================================

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(ttl=30)
def load_price_data(pair, limit=500):
    df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=limit)
    if df.empty:
        return df
    df = TechnicalAnalyzer.add_all_indicators(df)
    return df


@st.cache_data(ttl=30)
def get_signal(pair):
    model_artifact = load_model()
    if model_artifact is None:
        return None
    
    try:
        prices_df = load_price_data(pair, limit=500)
        if prices_df.empty:
            return None
        
        engineer = FeatureEngineer(prediction_horizon=24, threshold_multiplier=1.5)
        features_df, _ = engineer.build_dataset(prices_df)
        
        if features_df.empty:
            return None
        
        latest = features_df.iloc[-1:].copy()
        non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
        X = latest.drop(columns=[c for c in non_feature_cols if c in latest.columns])
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        expected_features = model_artifact['feature_names']
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        
        model = model_artifact['model']
        label_encoder = model_artifact['label_encoder']
        
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        prediction_decoded = label_encoder.inverse_transform([prediction])[0]
        label_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        
        return {
            'signal': label_map.get(prediction_decoded, 'HOLD'),
            'confidence': float(probabilities.max()),
            'price': float(prices_df['close'].iloc[-1]),
            'rsi': float(prices_df['RSI'].iloc[-1]) if 'RSI' in prices_df.columns else None,
            'macd': float(prices_df['MACD'].iloc[-1]) if 'MACD' in prices_df.columns else None,
            'all_probs': probabilities.tolist(),
        }
    except Exception:
        return None


def format_idr(amount):
    if amount >= 1_000_000_000:
        return f"Rp {amount/1_000_000_000:.3f}M"
    elif amount >= 1_000_000:
        return f"Rp {amount/1_000_000:.3f}jt"
    elif amount >= 1_000:
        return f"Rp {amount/1_000:.0f}rb"
    return f"Rp {amount:,.0f}"


def generate_order_book(current_price, depth=10):
    """Generate simulated order book."""
    bids = []
    asks = []
    np.random.seed(int(time.time() / 30))  # Change every 30s
    
    for i in range(1, depth + 1):
        bid_price = current_price * (1 - 0.001 * i)
        ask_price = current_price * (1 + 0.001 * i)
        bid_qty = np.random.uniform(0.1, 5.0)
        ask_qty = np.random.uniform(0.1, 5.0)
        
        bids.append({'price': bid_price, 'qty': bid_qty})
        asks.append({'price': ask_price, 'qty': ask_qty})
    
    return bids, asks


# ============================================================
# Header Bar
# ============================================================

selected_pair = st.session_state.selected_pair
info = PAIRS_INFO[selected_pair]

df = load_price_data(selected_pair, limit=24)
current_price = df['close'].iloc[-1] if not df.empty else 0
yesterday_price = df['close'].iloc[0] if not df.empty else 0
change_pct = ((current_price - yesterday_price) / yesterday_price * 100) if yesterday_price > 0 else 0

col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([2, 1.5, 1.2, 1.2, 1])

with col_h1:
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <span style='font-size: 36px; color: {info["color"]};'>{info["icon"]}</span>
        <div>
            <h2 style='margin: 0; color: white;'>{info['name']} / IDR</h2>
            <p style='margin: 0; color: #94a3b8; font-size: 12px;'>🇮🇩 Indodax Exchange</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    color = "#22c55e" if change_pct >= 0 else "#ef4444"
    st.markdown(f"""
    <div>
        <p style='margin: 0; color: #94a3b8; font-size: 11px;'>LAST PRICE</p>
        <h2 style='margin: 0; color: {color};'>{format_idr(current_price)}</h2>
        <p style='margin: 0; color: {color}; font-size: 14px;'>
            {'▲' if change_pct >= 0 else '▼'} {change_pct:+.2f}% (24h)
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_h3:
    if not df.empty:
        high_24h = df['high'].max()
        st.markdown(f"""
        <div>
            <p style='margin: 0; color: #94a3b8; font-size: 11px;'>24H HIGH</p>
            <h4 style='margin: 0; color: #22c55e;'>{format_idr(high_24h)}</h4>
        </div>
        """, unsafe_allow_html=True)

with col_h4:
    if not df.empty:
        low_24h = df['low'].min()
        st.markdown(f"""
        <div>
            <p style='margin: 0; color: #94a3b8; font-size: 11px;'>24H LOW</p>
            <h4 style='margin: 0; color: #ef4444;'>{format_idr(low_24h)}</h4>
        </div>
        """, unsafe_allow_html=True)

with col_h5:
    st.markdown(f"""
    <div style='text-align: right;'>
        <span class='live-indicator'>● LIVE</span>
        <p style='margin: 5px 0 0 0; color: #94a3b8; font-size: 11px;'>
            {datetime.now().strftime('%H:%M:%S')} WIB
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 15px 0; border-color: #2a3142;'>", unsafe_allow_html=True)


# ============================================================
# Main Layout: 3 Columns
# ============================================================

col_left, col_center, col_right = st.columns([1, 3, 1.3])


# ============================================================
# LEFT COLUMN: Watchlist
# ============================================================

with col_left:
    st.markdown("### 👁️ Watchlist")
    
    for pair, pair_info in PAIRS_INFO.items():
        df_pair = load_price_data(pair, limit=24)
        if df_pair.empty:
            continue
        
        price = df_pair['close'].iloc[-1]
        change = (price / df_pair['close'].iloc[0] - 1) * 100
        change_color = "#22c55e" if change >= 0 else "#ef4444"
        is_active = pair == st.session_state.selected_pair
        
        # Use button for clickability
        if st.button(
            f"{pair_info['symbol']}\n{format_idr(price)}\n{change:+.2f}%",
            key=f"watch_{pair}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.selected_pair = pair
            st.rerun()
    
    st.markdown("---")
    
    # Quick signals overview
    st.markdown("### 🎯 Live Signals")
    for pair, pair_info in PAIRS_INFO.items():
        sig = get_signal(pair)
        if sig:
            color = {'BUY': '#22c55e', 'SELL': '#ef4444', 'HOLD': '#6b7280'}.get(sig['signal'])
            arrow = {'BUY': '▲', 'SELL': '▼', 'HOLD': '●'}.get(sig['signal'])
            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; padding: 6px 10px;
                        background: #1a1f2e; border-radius: 6px; margin: 4px 0;'>
                <span style='color: white; font-weight: 600;'>{pair_info['symbol']}</span>
                <span style='color: {color}; font-weight: 700;'>{arrow} {sig['signal']}</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# CENTER COLUMN: Main Chart
# ============================================================

with col_center:
    # Timeframe selector
    tf_cols = st.columns([1, 1, 1, 1, 1, 4])
    timeframes = [('1H', 50), ('4H', 100), ('1D', 200), ('1W', 500), ('1M', 1000)]
    
    for i, (tf, _) in enumerate(timeframes):
        with tf_cols[i]:
            if st.button(tf, key=f"tf_{tf}", use_container_width=True,
                        type="primary" if st.session_state.timeframe == tf else "secondary"):
                st.session_state.timeframe = tf
                st.rerun()
    
    # Get data based on timeframe
    tf_to_limit = {'1H': 50, '4H': 100, '1D': 200, '1W': 500, '1M': 1000}
    chart_limit = tf_to_limit.get(st.session_state.timeframe, 200)
    df_chart = load_price_data(selected_pair, limit=chart_limit)
    
    if not df_chart.empty:
        # Main candlestick chart
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.65, 0.15, 0.20],
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=("", "Volume", "RSI"),
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_chart['timestamp'],
                open=df_chart['open'], high=df_chart['high'],
                low=df_chart['low'], close=df_chart['close'],
                name="Price",
                increasing=dict(line=dict(color='#22c55e'), fillcolor='#22c55e'),
                decreasing=dict(line=dict(color='#ef4444'), fillcolor='#ef4444'),
            ),
            row=1, col=1,
        )
        
        # Bollinger Bands
        if 'BB_upper' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart['timestamp'], y=df_chart['BB_upper'],
                           name='BB Upper', line=dict(color='rgba(167, 139, 250, 0.5)', width=1)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_chart['timestamp'], y=df_chart['BB_lower'],
                           name='BB Lower', line=dict(color='rgba(167, 139, 250, 0.5)', width=1),
                           fill='tonexty', fillcolor='rgba(167, 139, 250, 0.05)'),
                row=1, col=1,
            )
        
        # Moving averages
        if 'SMA_20' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart['timestamp'], y=df_chart['SMA_20'],
                           name='SMA 20', line=dict(color='#f59e0b', width=2)),
                row=1, col=1,
            )
        if 'SMA_50' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart['timestamp'], y=df_chart['SMA_50'],
                           name='SMA 50', line=dict(color='#a78bfa', width=2)),
                row=1, col=1,
            )
        
        # Volume
        colors = ['#22c55e' if c >= o else '#ef4444' 
                  for c, o in zip(df_chart['close'], df_chart['open'])]
        fig.add_trace(
            go.Bar(x=df_chart['timestamp'], y=df_chart['volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=2, col=1,
        )
        
        # RSI
        if 'RSI' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart['timestamp'], y=df_chart['RSI'],
                           name='RSI', line=dict(color='#a78bfa', width=2),
                           fill='tozeroy', fillcolor='rgba(167, 139, 250, 0.1)'),
                row=3, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.5)", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=3, col=1)
        
        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            paper_bgcolor='#0a0e1a',
            plot_bgcolor='#0a0e1a',
            font=dict(color='#e2e8f0'),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor='rgba(0,0,0,0)',
            ),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        
        fig.update_xaxes(gridcolor='rgba(42, 49, 66, 0.5)', showgrid=True)
        fig.update_yaxes(gridcolor='rgba(42, 49, 66, 0.5)', showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{selected_pair}")


# ============================================================
# RIGHT COLUMN: Trading Panel + Signal
# ============================================================

with col_right:
    # AI Signal Panel
    signal_data = get_signal(selected_pair)
    
    if signal_data:
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        
        signal_html = {
            'BUY': '<div class="signal-buy">▲ BUY SIGNAL</div>',
            'SELL': '<div class="signal-sell">▼ SELL SIGNAL</div>',
            'HOLD': '<div class="signal-hold">● HOLD POSITION</div>',
        }.get(signal)
        
        st.markdown("### 🤖 AI Signal")
        st.markdown(signal_html, unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown(f"""
        <div style='margin: 15px 0;'>
            <p style='color: #94a3b8; font-size: 11px; margin: 0;'>CONFIDENCE</p>
            <div class='indicator-bar' style='margin-top: 5px;'>
                <div class='indicator-fill' style='width: {confidence*100}%;'></div>
            </div>
            <p style='color: white; margin: 5px 0 0 0; font-size: 14px; text-align: right;'>
                <strong>{confidence*100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution
        if 'all_probs' in signal_data:
            probs = signal_data['all_probs']
            labels = ['SELL', 'HOLD', 'BUY']
            colors = ['#ef4444', '#6b7280', '#22c55e']
            
            st.markdown("**Probability Distribution:**")
            for label, prob, color in zip(labels, probs, colors):
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; 
                            margin: 5px 0;'>
                    <span style='color: {color}; font-weight: 700; width: 50px;'>{label}</span>
                    <div style='flex: 1; background: #2a3142; height: 18px; border-radius: 4px; 
                                margin: 0 10px; overflow: hidden;'>
                        <div style='width: {prob*100}%; height: 100%; background: {color};'></div>
                    </div>
                    <span style='color: white; font-size: 12px; width: 50px; text-align: right;'>
                        {prob*100:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Buy/Sell Action Buttons
    st.markdown("### 💰 Quick Actions")
    
    col_buy, col_sell = st.columns(2)
    
    with col_buy:
        if st.button("📈 BUY", key="buy_btn", use_container_width=True, type="primary"):
            st.session_state.show_modal = True
            st.session_state.modal_action = 'BUY'
    
    with col_sell:
        if st.button("📉 SELL", key="sell_btn", use_container_width=True, type="secondary"):
            st.session_state.show_modal = True
            st.session_state.modal_action = 'SELL'
    
    # Modal-like dialog
    if st.session_state.show_modal:
        action = st.session_state.modal_action
        action_color = '#22c55e' if action == 'BUY' else '#ef4444'
        
        st.markdown(f"""
        <div style='background: #1a1f2e; padding: 20px; border-radius: 10px; 
                    border: 2px solid {action_color}; margin: 15px 0;'>
            <h3 style='color: {action_color}; margin: 0;'>{action} Order</h3>
            <p style='color: #94a3b8; margin: 10px 0;'>
                Confirm {action} order for {info['name']} at <strong>{format_idr(current_price)}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        amount = st.number_input("Amount (IDR)", value=1000000, step=100000, key="order_amount")
        qty = amount / current_price if current_price > 0 else 0
        st.markdown(f"**Quantity:** {qty:.6f} {info['symbol']}")
        
        col_c, col_x = st.columns(2)
        with col_c:
            if st.button("✓ Confirm", key="confirm_order", type="primary", use_container_width=True):
                # Add to trade history
                st.session_state.trade_history.insert(0, {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'action': action,
                    'pair': info['symbol'],
                    'price': current_price,
                    'qty': qty,
                    'amount': amount,
                })
                st.session_state.show_modal = False
                st.success(f"✓ {action} order simulated! (Demo mode)")
                time.sleep(1)
                st.rerun()
        with col_x:
            if st.button("✗ Cancel", key="cancel_order", use_container_width=True):
                st.session_state.show_modal = False
                st.rerun()
    
    st.markdown("---")
    
    # Order Book (simulated)
    st.markdown("### 📋 Order Book")
    
    if current_price > 0:
        bids, asks = generate_order_book(current_price, depth=5)
        
        # Asks (sell orders) - shown in red
        for ask in reversed(asks):
            st.markdown(f"""
            <div class='order-row ask-row'>
                <span style='color: #ef4444;'>{format_idr(ask['price'])}</span>
                <span style='color: #94a3b8;'>{ask['qty']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Spread
        spread = asks[0]['price'] - bids[0]['price']
        spread_pct = (spread / bids[0]['price']) * 100
        st.markdown(f"""
        <div style='text-align: center; padding: 8px; background: #2a3142; 
                    border-radius: 4px; margin: 4px 0;'>
            <span style='color: white; font-weight: 700;'>Spread: {format_idr(spread)} ({spread_pct:.3f}%)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Bids (buy orders) - shown in green
        for bid in bids:
            st.markdown(f"""
            <div class='order-row bid-row'>
                <span style='color: #22c55e;'>{format_idr(bid['price'])}</span>
                <span style='color: #94a3b8;'>{bid['qty']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# Bottom Section: Tabs (Indicators, Trade History, Stats)
# ============================================================

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicators", "📜 Trade History", 
                                   "📈 Statistics", "ℹ️ About"])

with tab1:
    st.markdown("### Technical Indicators")
    
    if not df.empty:
        latest = df.iloc[-1]
        
        # Indicators in 4 columns
        ind_cols = st.columns(4)
        
        with ind_cols[0]:
            rsi = latest.get('RSI', 0)
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            rsi_color = "#ef4444" if rsi > 70 else "#22c55e" if rsi < 30 else "#6b7280"
            st.markdown(f"""
            <div class='stat-card'>
                <p class='stat-label'>RSI (14)</p>
                <p class='stat-value' style='color: {rsi_color};'>{rsi:.1f}</p>
                <p style='color: {rsi_color}; font-size: 11px; margin: 5px 0 0 0;'>{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with ind_cols[1]:
            macd = latest.get('MACD', 0)
            macd_status = "Bullish" if macd > 0 else "Bearish"
            macd_color = "#22c55e" if macd > 0 else "#ef4444"
            st.markdown(f"""
            <div class='stat-card'>
                <p class='stat-label'>MACD</p>
                <p class='stat-value' style='color: {macd_color};'>{macd:.2f}</p>
                <p style='color: {macd_color}; font-size: 11px; margin: 5px 0 0 0;'>{macd_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with ind_cols[2]:
            bb_pos = latest.get('BB_position', 0.5)
            bb_status = "Upper Zone" if bb_pos > 0.8 else "Lower Zone" if bb_pos < 0.2 else "Mid Zone"
            bb_color = "#ef4444" if bb_pos > 0.8 else "#22c55e" if bb_pos < 0.2 else "#6b7280"
            st.markdown(f"""
            <div class='stat-card'>
                <p class='stat-label'>BB Position</p>
                <p class='stat-value' style='color: {bb_color};'>{bb_pos:.2f}</p>
                <p style='color: {bb_color}; font-size: 11px; margin: 5px 0 0 0;'>{bb_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with ind_cols[3]:
            adx = latest.get('ADX', 0)
            adx_status = "Strong Trend" if adx > 25 else "Weak Trend"
            adx_color = "#22c55e" if adx > 25 else "#6b7280"
            st.markdown(f"""
            <div class='stat-card'>
                <p class='stat-label'>ADX (14)</p>
                <p class='stat-value' style='color: {adx_color};'>{adx:.1f}</p>
                <p style='color: {adx_color}; font-size: 11px; margin: 5px 0 0 0;'>{adx_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # More indicators
        st.markdown("<br>", unsafe_allow_html=True)
        
        more_cols = st.columns(4)
        with more_cols[0]:
            stoch_k = latest.get('STOCH_K', 0)
            st.markdown(f"<div class='stat-card'><p class='stat-label'>Stochastic K</p><p class='stat-value'>{stoch_k:.1f}</p></div>", unsafe_allow_html=True)
        with more_cols[1]:
            mfi = latest.get('MFI', 0)
            st.markdown(f"<div class='stat-card'><p class='stat-label'>MFI</p><p class='stat-value'>{mfi:.1f}</p></div>", unsafe_allow_html=True)
        with more_cols[2]:
            atr = latest.get('ATR', 0)
            st.markdown(f"<div class='stat-card'><p class='stat-label'>ATR</p><p class='stat-value'>{atr:,.0f}</p></div>", unsafe_allow_html=True)
        with more_cols[3]:
            cci = latest.get('CCI', 0)
            st.markdown(f"<div class='stat-card'><p class='stat-label'>CCI</p><p class='stat-value'>{cci:.1f}</p></div>", unsafe_allow_html=True)


with tab2:
    st.markdown("### Trade History (Demo)")
    
    if st.session_state.trade_history:
        # Header
        st.markdown("""
        <div class='trade-row' style='background: #1a1f2e; font-weight: 700; color: #94a3b8;'>
            <span>TIME</span>
            <span>SIDE</span>
            <span>PRICE</span>
            <span>QTY / AMOUNT</span>
        </div>
        """, unsafe_allow_html=True)
        
        for trade in st.session_state.trade_history[:20]:
            side_class = 'trade-buy' if trade['action'] == 'BUY' else 'trade-sell'
            arrow = '▲' if trade['action'] == 'BUY' else '▼'
            
            st.markdown(f"""
            <div class='trade-row'>
                <span style='color: #94a3b8;'>{trade['time']}</span>
                <span class='{side_class}'>{arrow} {trade['action']}</span>
                <span style='color: white;'>{format_idr(trade['price'])}</span>
                <span style='color: #94a3b8;'>{trade['qty']:.4f} / {format_idr(trade['amount'])}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Clear History"):
            st.session_state.trade_history = []
            st.rerun()
    else:
        st.info("No trades yet. Click BUY or SELL to simulate trades!")


with tab3:
    st.markdown("### Market Statistics")
    
    if not df.empty:
        df_full = load_price_data(selected_pair, limit=1000)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            volume_24h = df['volume'].sum()
            st.markdown(f"""
            <div class='stat-card'>
                <p class='stat-label'>24h Volume</p>
                <p class='stat-value' style='color: #a78bfa;'>{format_idr(volume_24h)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            volatility = df['close'].pct_change().std() * 100
            st.markdown(f"""
            <div class='stat-card'>
                <p class='stat-label'>Volatility (24h)</p>
                <p class='stat-value' style='color: #f59e0b;'>{volatility:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            if not df_full.empty:
                all_time_high = df_full['high'].max()
                st.markdown(f"""
                <div class='stat-card'>
                    <p class='stat-label'>All-Time High</p>
                    <p class='stat-value' style='color: #22c55e;'>{format_idr(all_time_high)}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_s4:
            if not df_full.empty:
                all_time_low = df_full['low'].min()
                st.markdown(f"""
                <div class='stat-card'>
                    <p class='stat-label'>All-Time Low</p>
                    <p class='stat-value' style='color: #ef4444;'>{format_idr(all_time_low)}</p>
                </div>
                """, unsafe_allow_html=True)


with tab4:
    st.markdown("""
    ### About This Dashboard
    
    Professional trading dashboard inspired by TradingView dan Binance, dibangun dengan:
    
    - **Streamlit** - Web framework
    - **Plotly** - Interactive charts
    - **XGBoost** - ML predictions
    - **CCXT** - Exchange data
    - **Custom CSS** - Trading-pro UI
    
    ### Features
    
    - 🤖 AI-powered signals dengan confidence scoring
    - 📊 Interactive candlestick charts
    - 📋 Live order book simulation
    - 💰 Trade simulator (demo mode)
    - 📈 10+ technical indicators
    - 📜 Trade history tracking
    
    ### ⚠️ Disclaimer
    
    Dashboard ini adalah **portfolio demo** dan **bukan platform trading sungguhan**.
    Semua "trade" hanya simulasi untuk demonstrasi UI/UX. Tidak ada uang asli yang ditransaksikan.
    
    **Untuk trading real**, gunakan platform resmi seperti Indodax, Binance, dll.
    """)