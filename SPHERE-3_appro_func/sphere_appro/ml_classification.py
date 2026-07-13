"""ML classification pipeline for SPHERE-3 particle type identification."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
    'R_ch', 'sw', 'x0', 'y0',
    'chi2_ndf', 'mean_abs_d', 'max_abs_d',
    'Rc_snow', 'I_max', 'sum',
    'Int', 'err_Int',
]

SHAPE_FEATURE_COLUMNS = [
    'p1', 'p2', 'p3',      # core shape
    'p5', 'p6',             # tail shape
    'R_ch', 'sw',           # sigmoid transition
    'x0', 'y0',             # shower center (detector)
    'chi2_ndf',             # fit quality (dimensionless)
    'Rc_snow',              # shower center (snow)
]

PARTICLE_ENCODING = {'p': 0, 'N': 1, 'Fe': 2}
PARTICLE_DECODING = {v: k for k, v in PARTICLE_ENCODING.items()}

META_COLUMNS = ['particle', 'energy', 'angle', 'height']

MIN_EVENTS_PER_CLASS_SLICE = 50


def load_and_split(parquet_path, test_size=0.2, random_state=42, feature_columns=None):
    """Load parquet, encode target, stratified split."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    df = pd.read_parquet(parquet_path)
    numeric = df.select_dtypes(include='number')
    assert numeric.isna().sum().sum() == 0, 'Input contains NaN values'
    assert not np.isinf(numeric.values).any(), 'Input contains inf values'

    X = df[feature_columns].values
    y = df['particle'].map(PARTICLE_ENCODING).values

    strata = df['particle'] + '_' + df['energy'] + '_' + df['angle'].astype(str) + '_' + df['height'].astype(str)

    # Split X, y, and df.index together to track test set metadata
    X_train, X_test, y_train, y_test, _, meta_idx = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state,
        stratify=strata,
    )

    meta_test = df.loc[meta_idx, META_COLUMNS].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, meta_test


def train_and_evaluate(X_train, X_test, y_train, y_test, meta_test, output_dir,
                       feature_columns=None):
    """Train RF/XGBoost/LightGBM, evaluate, save results."""
    import joblib
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        ),
        'xgboost': XGBClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, verbose=-1,
        ),
    }

    predictions = {}
    overall_metrics = {}
    feature_importances = {}

    for name, model in models.items():
        logger.info('Training %s...', name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        overall_metrics[name] = {'accuracy': round(acc, 4), 'f1_macro': round(f1, 4)}
        logger.info('%s: accuracy=%.4f, f1_macro=%.4f', name, acc, f1)

        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = {
                col: round(float(imp), 6)
                for col, imp in zip(feature_columns, model.feature_importances_)
            }

        joblib.dump(model, models_dir / f'{name}.joblib')

    per_slice = _compute_per_slice_metrics(y_test, predictions, meta_test)

    metrics = {
        'overall': overall_metrics,
        'per_slice': per_slice,
        'feature_importances': feature_importances,
    }

    with open(output_dir / 'ml_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    result_df = meta_test.copy()
    result_df['y_true'] = y_test
    for name, preds in predictions.items():
        result_df[f'y_pred_{name}'] = preds
    result_df.to_parquet(output_dir / 'ml_results.parquet', index=False)

    logger.info('Results saved to %s', output_dir)
    return metrics


def _compute_per_slice_metrics(y_test, predictions, meta_test):
    """Compute accuracy per (energy, angle, height) slice."""
    result = []
    meta_df = meta_test.copy()
    meta_df['y_true'] = y_test
    for name, preds in predictions.items():
        meta_df[f'pred_{name}'] = preds

    for (energy, angle, height), group in meta_df.groupby(['energy', 'angle', 'height']):
        class_counts = group['y_true'].value_counts()
        if class_counts.min() < MIN_EVENTS_PER_CLASS_SLICE:
            continue

        entry = {'energy': energy, 'angle': int(angle), 'height': int(height)}
        for name in predictions:
            acc = accuracy_score(group['y_true'], group[f'pred_{name}'])
            entry[name] = {'accuracy': round(acc, 4)}
        result.append(entry)

    return result


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='ML classification for SPHERE-3')
    parser.add_argument('parquet_path', help='Path to results.parquet')
    parser.add_argument('-o', '--output-dir', default='analysis_output',
                        help='Output directory (default: analysis_output)')
    parser.add_argument('--shape-only', action='store_true',
                        help='Use only shape features (no absolute amplitudes)')
    args = parser.parse_args()

    feat_cols = SHAPE_FEATURE_COLUMNS if args.shape_only else FEATURE_COLUMNS
    logger.info('Features (%d): %s', len(feat_cols), feat_cols)

    X_train, X_test, y_train, y_test, meta_test = load_and_split(
        args.parquet_path, feature_columns=feat_cols,
    )
    logger.info('Split: %d train, %d test', len(X_train), len(X_test))
    metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, meta_test, args.output_dir,
        feature_columns=feat_cols,
    )
    logger.info('Overall accuracies: %s',
                {k: v['accuracy'] for k, v in metrics['overall'].items()})
