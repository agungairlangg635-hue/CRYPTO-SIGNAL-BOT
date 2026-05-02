"""
XGBoost Training dengan Per-Pair Train-Test Split.
Fixes: BTC test data won't leak from BTC train data (proper validation).
"""

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


TRAINING_DATA_PATH = Path('data/processed/features/all_pairs_training_data.csv')
MODEL_OUTPUT_DIR = Path('src/models/saved')
CHARTS_OUTPUT_DIR = Path('docs/images')
TEST_SIZE_PER_PAIR = 0.2  # 20% per pair untuk test

XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.0,
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'mlogloss',
}


def load_training_data():
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")
    df = pd.read_csv(TRAINING_DATA_PATH)
    logger.info(f"Loaded training data: {len(df)} rows, {df.shape[1]} columns")
    return df


def split_per_pair(df, test_size=0.2):
    """
    PROPER split: 20% terakhir DARI SETIAP PAIR untuk test.
    Bukan 20% terakhir combined dataset (yang bias ke XRP).
    """
    train_dfs = []
    test_dfs = []
    
    for pair in df['pair'].unique():
        pair_df = df[df['pair'] == pair].copy().reset_index(drop=True)
        
        # 80% awal = train, 20% akhir = test (PER PAIR)
        split_idx = int(len(pair_df) * (1 - test_size))
        
        train_dfs.append(pair_df.iloc[:split_idx])
        test_dfs.append(pair_df.iloc[split_idx:])
        
        logger.info(
            f"  {pair}: {len(pair_df)} total | "
            f"train={split_idx} | test={len(pair_df) - split_idx}"
        )
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df


def prepare_features(df):
    non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])
    
    return X, y


def train_model(train_df, test_df):
    logger.info("Starting model training (per-pair split)")
    
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    
    # Align columns (in case of mismatch)
    X_test = X_test[X_train.columns]
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)
    
    logger.info(f"Target classes: {label_encoder.classes_.tolist()}")
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    sample_weights = compute_sample_weight('balanced', y_train_encoded)
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS, num_class=n_classes)
    model.fit(
        X_train, y_train_encoded,
        sample_weight=sample_weights,
        verbose=False,
    )
    
    logger.info("Training complete")
    return model, label_encoder, X_test, y_test_encoded, X_train.columns.tolist()


def evaluate_model(model, label_encoder, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy * 100:.2f}%")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_decoded, y_pred_decoded, labels=[-1, 0, 1], zero_division=0,
    )
    
    label_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    for i, label in enumerate([-1, 0, 1]):
        name = label_names[label]
        logger.info(
            f"  {name}: precision={precision[i]:.3f}, "
            f"recall={recall[i]:.3f}, f1={f1[i]:.3f}, support={support[i]}"
        )
    
    max_proba = y_proba.max(axis=1)
    logger.info(f"Confidence: mean={max_proba.mean():.3f}, max={max_proba.max():.3f}")
    
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=[-1, 0, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm,
    }


def evaluate_per_pair(model, label_encoder, test_df, feature_names):
    """Evaluate accuracy per pair untuk verify no leakage."""
    logger.info("")
    logger.info("Per-pair test accuracy (TRUE out-of-sample):")
    
    pairs = test_df['pair'].unique()
    pair_accs = {}
    
    for pair in pairs:
        pair_df = test_df[test_df['pair'] == pair]
        X_pair, y_pair = prepare_features(pair_df)
        X_pair = X_pair[feature_names]
        
        y_pair_encoded = label_encoder.transform(y_pair)
        y_pred = model.predict(X_pair)
        
        acc = accuracy_score(y_pair_encoded, y_pred)
        pair_accs[pair] = acc
        logger.info(f"  {pair:<12} accuracy={acc*100:5.2f}% (n={len(pair_df)})")
    
    return pair_accs


def save_charts(model, feature_names, cm):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df.plot(
        kind='barh', x='feature', y='importance', ax=ax,
        color='steelblue', legend=False,
    )
    ax.set_title('Top 20 Most Important Features')
    ax.invert_yaxis()
    plt.tight_layout()
    
    CHARTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_OUTPUT_DIR / 'feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['SELL', 'HOLD', 'BUY'],
        yticklabels=['SELL', 'HOLD', 'BUY'], ax=ax,
    )
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(CHARTS_OUTPUT_DIR / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()


def save_model(model, label_encoder, feature_names, metrics):
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model_artifact = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'metrics': metrics,
        'training_date': pd.Timestamp.now().isoformat(),
        'xgboost_params': XGBOOST_PARAMS,
        'split_method': 'per_pair_temporal',
    }
    
    output_path = MODEL_OUTPUT_DIR / 'xgboost_signal_model.pkl'
    joblib.dump(model_artifact, output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    try:
        df = load_training_data()
        
        logger.info("Splitting train/test PER PAIR...")
        train_df, test_df = split_per_pair(df, test_size=TEST_SIZE_PER_PAIR)
        
        model, label_encoder, X_test, y_test, feature_names = train_model(
            train_df, test_df
        )
        
        metrics = evaluate_model(model, label_encoder, X_test, y_test)
        evaluate_per_pair(model, label_encoder, test_df, feature_names)
        
        save_charts(model, feature_names, metrics['confusion_matrix'])
        save_model(model, label_encoder, feature_names, metrics)
        
        logger.info("=" * 60)
        logger.info(f"FINAL ACCURACY: {metrics['accuracy'] * 100:.2f}%")
        logger.info("=" * 60)
    
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()