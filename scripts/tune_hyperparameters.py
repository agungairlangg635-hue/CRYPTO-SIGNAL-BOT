"""
Hyperparameter Tuning untuk XGBoost Signal Model.

Pakai Optuna untuk Bayesian optimization — lebih efisien
dari grid search atau random search.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


TRAINING_DATA_PATH = Path('data/processed/features/all_pairs_training_data.csv')
BEST_PARAMS_PATH = Path('src/models/saved/best_xgboost_params.pkl')
N_TRIALS = 50  # Number of optimization trials


def load_and_prepare_data():
    """Load data dan prepare untuk tuning."""
    df = pd.read_csv(TRAINING_DATA_PATH)
    
    non_feature_cols = ['target', 'pair', 'signal', 'signal_strength']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])
    
    # Encode targets
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y_encoded, label_encoder


def objective(trial, X, y, n_classes):
    """
    Optuna objective function — return F1 score yang akan di-maximize.
    Pakai TimeSeriesSplit (5-fold) untuk honest cross-validation.
    """
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        
        # Hyperparameters yang akan di-tune
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        sample_weights = compute_sample_weight('balanced', y_train)
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        # Macro F1 untuk multi-class (treats all classes equally)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    return mean_f1


def main():
    logger.info("=" * 65)
    logger.info("HYPERPARAMETER TUNING dengan OPTUNA")
    logger.info("=" * 65)
    logger.info(f"Trials: {N_TRIALS}")
    logger.info(f"Method: TimeSeriesSplit 5-fold + Bayesian Optimization")
    logger.info("")
    
    # Load data
    X, y_encoded, label_encoder = load_and_prepare_data()
    n_classes = len(label_encoder.classes_)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_signal_tuning',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    # Optimize
    logger.info("Starting optimization...")
    
    def callback(study, trial):
        if trial.number % 10 == 0:
            logger.info(
                f"Trial {trial.number}/{N_TRIALS} | "
                f"Best F1: {study.best_value:.4f}"
            )
    
    study.optimize(
        lambda trial: objective(trial, X, y_encoded, n_classes),
        n_trials=N_TRIALS,
        callbacks=[callback],
        show_progress_bar=False,
    )
    
    # Results
    logger.info("=" * 65)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Best F1 Score: {study.best_value:.4f}")
    logger.info(f"Best Trial: #{study.best_trial.number}")
    logger.info("")
    logger.info("Best Hyperparameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            logger.info(f"  {param:<25} {value:.4f}")
        else:
            logger.info(f"  {param:<25} {value}")
    
    # Save best params
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': N_TRIALS,
        'study': study,
    }, BEST_PARAMS_PATH)
    
    logger.info("")
    logger.info(f"Best params saved to: {BEST_PARAMS_PATH}")
    logger.info("Next step: re-train model dengan params ini")


if __name__ == "__main__":
    main()