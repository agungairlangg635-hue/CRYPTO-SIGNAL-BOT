"""Run Backtest dengan Adaptive Confidence Threshold."""

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.technical import TechnicalAnalyzer
from src.database.repository import PriceRepository
from src.models.backtester import Backtester
from src.models.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path('src/models/saved/xgboost_signal_model.pkl')
CHARTS_DIR = Path('docs/images')


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model trained on {artifact['training_date']}")
    return artifact


def generate_signals(model_artifact, pair):
    logger.info(f"Generating signals for {pair}")
    
    prices_df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=10000)
    if prices_df.empty:
        raise ValueError(f"No data for {pair}")
    
    prices_df = TechnicalAnalyzer.add_all_indicators(prices_df)
    
    engineer = FeatureEngineer(prediction_horizon=24, threshold_multiplier=1.5)
    features_df, _ = engineer.build_dataset(prices_df)
    
    model = model_artifact['model']
    label_encoder = model_artifact['label_encoder']
    expected_features = model_artifact['feature_names']
    
    test_start_idx = int(len(features_df) * 0.8)
    test_features = features_df.iloc[test_start_idx:].copy()
    
    logger.info(f"Out-of-sample backtest: {len(test_features)} candles")
    
    X = test_features.copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    predictions_decoded = label_encoder.inverse_transform(predictions)
    confidences = probabilities.max(axis=1)
    
    label_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    signal_labels = [label_map[p] for p in predictions_decoded]
    
    signals_df = pd.DataFrame({
        'signal': signal_labels,
        'confidence': confidences,
    })
    
    valid_indices = test_features.index
    prices_aligned = prices_df.loc[valid_indices].reset_index(drop=True)
    
    n_buy = sum(s == 'BUY' for s in signal_labels)
    n_sell = sum(s == 'SELL' for s in signal_labels)
    n_hold = sum(s == 'HOLD' for s in signal_labels)
    
    # Diagnostic: confidence distribution
    logger.info(
        f"Signals: BUY={n_buy}, HOLD={n_hold}, SELL={n_sell}"
    )
    logger.info(
        f"Confidence: mean={confidences.mean():.3f}, "
        f"max={confidences.max():.3f}, "
        f">0.40 = {(confidences > 0.40).sum()}, "
        f">0.45 = {(confidences > 0.45).sum()}"
    )
    
    return prices_aligned, signals_df


def print_results_report(results, benchmark, pair):
    logger.info("=" * 65)
    logger.info(f"BACKTEST RESULTS: {pair.upper()}")
    logger.info("=" * 65)
    logger.info(f"  Initial:        Rp {results.initial_capital:>15,.0f}")
    logger.info(f"  Final:          Rp {results.final_capital:>15,.0f}")
    logger.info(f"  Return %:          {results.total_return_pct:>14.2f}%")
    logger.info(f"  Total Trades:      {results.n_trades:>14}")
    logger.info(f"  Win Rate:          {results.win_rate:>13.2f}%")
    logger.info(f"  Sharpe Ratio:      {results.sharpe_ratio:>14.2f}")
    logger.info(f"  Max DD %:          {results.max_drawdown_pct:>13.2f}%")
    logger.info(f"  Profit Factor:     {results.profit_factor:>14.2f}")
    logger.info(f"  B&H Return:        {benchmark['total_return_pct']:>13.2f}%")
    
    alpha = results.total_return_pct - benchmark['total_return_pct']
    if alpha > 0:
        logger.info(f"  Alpha:             +{alpha:>12.2f}%  [BEAT MARKET]")
    else:
        logger.info(f"  Alpha:             {alpha:>13.2f}%  [UNDERPERFORM]")


def plot_equity_curve(results, benchmark, pair):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True,
    )
    
    equity = results.equity_curve
    ax1.plot(equity['timestamp'], equity['equity'], label='Strategy',
             color='steelblue', linewidth=2)
    ax1.axhline(results.initial_capital, color='gray', linestyle='--',
                alpha=0.5, label='Initial Capital')
    bh_final = benchmark['final_capital']
    ax1.axhline(bh_final, color='orange', linestyle=':', alpha=0.7,
                label=f"Buy & Hold ({benchmark['total_return_pct']:+.1f}%)")
    
    ax1.set_title(f'Backtest Equity Curve: {pair.upper()}', fontweight='bold')
    ax1.set_ylabel('Equity (IDR)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(equity['timestamp'], equity['drawdown_pct'], 0,
                     color='red', alpha=0.3)
    ax2.plot(equity['timestamp'], equity['drawdown_pct'], color='darkred')
    ax2.set_title('Drawdown (%)', fontsize=10)
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / f'backtest_{pair}.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    try:
        model_artifact = load_model()
        
        # LOWER threshold to 0.40 — model konservatif butuh threshold lebih rendah
        backtester = Backtester(
            initial_capital=10_000_000,
            position_size_pct=0.20,
            taker_fee=0.003,
            confidence_threshold=0.40,  # LOWERED from 0.55
        )
        
        pairs = ['btc_idr', 'eth_idr', 'bnb_idr', 'sol_idr', 'xrp_idr']
        all_results = {}
        
        for pair in pairs:
            try:
                prices, signals = generate_signals(model_artifact, pair)
                results = backtester.run(prices, signals)
                benchmark = backtester.calculate_buy_and_hold(prices)
                
                print_results_report(results, benchmark, pair)
                plot_equity_curve(results, benchmark, pair)
                
                all_results[pair] = {'results': results, 'benchmark': benchmark}
            except Exception as e:
                logger.error(f"Backtest failed for {pair}: {e}")
                continue
        
        if all_results:
            logger.info("=" * 65)
            logger.info("BACKTEST SUMMARY")
            logger.info("=" * 65)
            
            total_strategy = sum(r['results'].total_return_pct for r in all_results.values()) / len(all_results)
            total_benchmark = sum(r['benchmark']['total_return_pct'] for r in all_results.values()) / len(all_results)
            avg_sharpe = sum(r['results'].sharpe_ratio for r in all_results.values()) / len(all_results)
            avg_win_rate = sum(r['results'].win_rate for r in all_results.values()) / len(all_results)
            total_trades = sum(r['results'].n_trades for r in all_results.values())
            
            logger.info(f"Average Strategy Return: {total_strategy:+.2f}%")
            logger.info(f"Average B&H Return:      {total_benchmark:+.2f}%")
            logger.info(f"Average Alpha:           {total_strategy - total_benchmark:+.2f}%")
            logger.info(f"Average Sharpe:          {avg_sharpe:.2f}")
            logger.info(f"Average Win Rate:        {avg_win_rate:.2f}%")
            logger.info(f"Total Trades:            {total_trades}")
    
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()