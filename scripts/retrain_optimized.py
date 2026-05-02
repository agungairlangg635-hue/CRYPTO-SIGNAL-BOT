"""
Re-train XGBoost model dengan hyperparameters yang udah di-tune.
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
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


TRAINING_DATA_PATH = Path('data/processed/features/all_pairs_training_data.csv')
BEST_PARAMS_PATH = Path('src/models/saved/best_xgboost_params.pkl')
MODEL_OUTPUT_PATH = Path('src/models/saved/xgboost_signal_model.pkl')
CHARTS_DIR = Path('docs/images')


def main():
    logger.info("=" * 65)
    logger.info("RE-TRAIN MODEL DENGAN OPTIMIZED PARAMETERS")
    logger.info("=" * 65)
    
    # Load best params
    if not BEST_PARAMS_PATH.exists():
        logger.error("Best params not found. Run tune_hyperparameters.py first")
        sys.exit(1)
    
    tuning_result = joblib.load(BEST_PARAMS_PATH)
    best_params = tuning_result['best_params']
    
    logger.info(f"Loaded best params (F1: {tuning_result['best_score']:.4f})")
    
    # Load data
    df = pd.read_csv(TRAINING_DATA_PATH)
    non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    
    # Temporal split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
    
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train dengan best params
    sample_weights = compute_sample_weight('balanced', y_train)
    
    final_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'early_stopping_rounds': 30,
        **best_params,  # Inject best params
    }
    
    logger.info("Training model dengan optimized parameters...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    logger.info(f"Training complete (best iteration: {model.best_iteration})")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("=" * 65)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    
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
    
    # Confusion matrix
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=[-1, 0, 1])
    
    # Save updated charts
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['SELL', 'HOLD', 'BUY'],
        yticklabels=['SELL', 'HOLD', 'BUY'], ax=ax,
    )
    ax.set_title('Confusion Matrix (Tuned Model)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df.plot(
        kind='barh', x='feature', y='importance', ax=ax,
        color='steelblue', legend=False,
    )
    ax.set_title('Top 20 Feature Importance (Tuned Model)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Charts updated in {CHARTS_DIR}")
    
    # Save model
    model_artifact = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist(),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
        },
        'training_date': pd.Timestamp.now().isoformat(),
        'xgboost_params': final_params,
        'tuning_score': tuning_result['best_score'],
    }
    
    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)
    
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
    logger.info("=" * 65)
    logger.info(f"FINAL ACCURACY: {accuracy * 100:.2f}%")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()