"""Diagnostic script untuk detect data leakage & overfitting."""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.technical import TechnicalAnalyzer
from src.database.repository import PriceRepository
from src.models.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 65)
    logger.info("MODEL DIAGNOSTIC - DETECT DATA LEAKAGE")
    logger.info("=" * 65)
    
    # Load model
    model_artifact = joblib.load('src/models/saved/xgboost_signal_model.pkl')
    model = model_artifact['model']
    label_encoder = model_artifact['label_encoder']
    expected_features = model_artifact['feature_names']
    
    # ====== TEST 1: Per-Pair Out-of-Sample ======
    logger.info("")
    logger.info("TEST 1: Per-pair out-of-sample accuracy")
    logger.info("-" * 65)
    
    pairs = ['btc_idr', 'eth_idr', 'bnb_idr', 'sol_idr', 'xrp_idr']
    engineer = FeatureEngineer(prediction_horizon=24, threshold_multiplier=1.5)
    
    results = {}
    for pair in pairs:
        prices_df = PriceRepository.get_ohlcv_df(pair, resolution='1h', limit=1000)
        prices_df = TechnicalAnalyzer.add_all_indicators(prices_df)
        features_df, _ = engineer.build_dataset(prices_df)
        
        # Out-of-sample: 20% terakhir
        test_start = int(len(features_df) * 0.8)
        test_df = features_df.iloc[test_start:].copy()
        
        # Prepare features
        non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
        X = test_df.drop(columns=[c for c in non_feature_cols if c in test_df.columns])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        
        y_true = test_df['target'].values
        y_true_encoded = label_encoder.transform(y_true)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        accuracy = accuracy_score(y_true_encoded, y_pred)
        confidence = y_proba.max(axis=1).mean()
        
        results[pair] = {
            'accuracy': accuracy,
            'confidence': confidence,
            'n_samples': len(test_df),
        }
        
        logger.info(
            f"{pair.upper():12} | accuracy={accuracy*100:5.2f}% | "
            f"confidence={confidence:.3f} | samples={len(test_df)}"
        )
    
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    avg_conf = np.mean([r['confidence'] for r in results.values()])
    logger.info(f"{'AVERAGE':12} | accuracy={avg_acc*100:5.2f}% | confidence={avg_conf:.3f}")
    
    # ====== TEST 2: Random Shuffle Test ======
    logger.info("")
    logger.info("TEST 2: Random shuffle test (should drop accuracy)")
    logger.info("-" * 65)
    
    df_combined = pd.read_csv('data/processed/features/all_pairs_training_data.csv')
    
    non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
    X_all = df_combined.drop(columns=[c for c in non_feature_cols if c in df_combined.columns])
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_all = X_all.select_dtypes(include=[np.number])
    
    for col in expected_features:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[expected_features]
    
    y_all = df_combined['target'].values
    y_all_encoded = label_encoder.transform(y_all)
    
    # Shuffle target — model harusnya FAIL di sini (50% accuracy max kalau 3 kelas = 33%)
    np.random.seed(42)
    y_shuffled = np.random.permutation(y_all_encoded)
    
    y_pred_shuffled = model.predict(X_all)
    acc_shuffled = accuracy_score(y_shuffled, y_pred_shuffled)
    acc_real = accuracy_score(y_all_encoded, y_pred_shuffled)
    
    logger.info(f"Real labels accuracy:     {acc_real*100:.2f}%")
    logger.info(f"Shuffled labels accuracy: {acc_shuffled*100:.2f}%")
    
    if acc_shuffled > 0.40:
        logger.info("VERDICT: Model SUSPICIOUSLY GOOD even on shuffled labels = LEAKAGE")
    elif acc_real - acc_shuffled > 0.20:
        logger.info("VERDICT: Model genuinely learned patterns")
    else:
        logger.info("VERDICT: Marginal improvement over random")
    
    # ====== TEST 3: Confidence Calibration ======
    logger.info("")
    logger.info("TEST 3: Confidence calibration (well-calibrated?)")
    logger.info("-" * 65)
    
    # Bin predictions by confidence, check if accuracy matches confidence
    y_proba_all = model.predict_proba(X_all)
    confidences = y_proba_all.max(axis=1)
    y_pred_all = model.predict(X_all)
    
    bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    logger.info(f"{'Conf range':<15} {'Avg Conf':<10} {'Accuracy':<10} {'Samples':<10}")
    for low, high in bins:
        mask = (confidences >= low) & (confidences < high)
        n = mask.sum()
        if n > 0:
            avg_conf = confidences[mask].mean()
            acc = accuracy_score(y_all_encoded[mask], y_pred_all[mask])
            indicator = ""
            if abs(avg_conf - acc) > 0.15:
                indicator = " <-- MISCALIBRATED"
            logger.info(f"{low:.2f}-{high:.2f}      {avg_conf:.3f}      {acc:.3f}      {n:>5}{indicator}")
    
    # ====== TEST 4: Feature Importance Top 10 ======
    logger.info("")
    logger.info("TEST 4: Top 10 features (cek kalau ada yang suspicious)")
    logger.info("-" * 65)
    
    importance_df = pd.DataFrame({
        'feature': expected_features,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(10)
    
    suspicious_keywords = ['future', 'next', 'forward', 'target', 'label']
    for _, row in importance_df.iterrows():
        suspicious = any(kw in row['feature'].lower() for kw in suspicious_keywords)
        flag = " <-- SUSPICIOUS" if suspicious else ""
        logger.info(f"  {row['feature']:<35} {row['importance']:.4f}{flag}")
    
    logger.info("")
    logger.info("=" * 65)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()