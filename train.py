"""Reproducible offline experiment. The test set never selects the threshold."""
from __future__ import annotations
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ids.schema import FEATURES, read_nsl, validate_features
from ids.pipeline import make_preprocessor, transform_frame, reconstruction_errors, sha256
from ids.metrics import calibrate_threshold, evaluate_scores, binary_metrics

ROOT = Path(__file__).resolve().parent


def split_indices(labels, seed=42):
    indices = np.arange(len(labels))
    train, heldout = train_test_split(indices, test_size=.2, stratify=labels, random_state=seed)
    validation, calibration = train_test_split(
        heldout, test_size=.5, stratify=np.asarray(labels)[heldout], random_state=seed)
    return train, validation, calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--version', default='ae_v2')
    args = parser.parse_args()
    if not args.version.replace('_', '').isalnum():
        parser.error('Version must contain letters, digits or underscore.')
    target = ROOT / 'models' / args.version
    if target.exists():
        parser.error('Model directory already exists; choose a new --version.')
    if args.epochs < 1:
        parser.error('--epochs must be positive.')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    source = ROOT / 'data/raw/KDDTrain+.txt'
    train_data = read_nsl(source)
    y = train_data.True_Class.to_numpy()
    ti, vi, ci = split_indices(y)
    normal_t = ti[y[ti] == 0]
    normal_v = vi[y[vi] == 0]
    normal_c = ci[y[ci] == 0]
    pre = make_preprocessor()
    x_train = np.asarray(pre.fit_transform(validate_features(train_data.iloc[normal_t])), dtype=np.float32)
    x_val, _ = transform_frame(pre, train_data.iloc[normal_v])
    x_cal, _ = transform_frame(pre, train_data.iloc[normal_c])
    width = x_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(width,)), tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(.2), tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(.2), tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(width, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='mse')
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train)).shuffle(
        len(x_train), seed=42, reshuffle_each_iteration=True).batch(256).with_options(options)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, x_val)).batch(256).with_options(options)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=3, restore_best_weights=True)], verbose=2)
    cal_scores = reconstruction_errors(model, x_cal)
    threshold = calibrate_threshold(cal_scores, .95)
    # All modeling and threshold decisions finish before KDDTest is opened.
    baseline_pre = make_preprocessor()
    bx_train = baseline_pre.fit_transform(validate_features(train_data.iloc[ti]))
    baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)
    baseline.fit(bx_train, y[ti])
    target.mkdir(parents=True)
    model.save(target / 'model.keras')
    joblib.dump(pre, target / 'preprocessor.joblib')
    np.savez_compressed(target / 'split_indices.npz', train=ti, validation=vi, calibration=ci)
    manifest = {
        'schema_version': 1, 'version': args.version, 'features': FEATURES,
        'encoded_features': width, 'parameters': model.count_params(), 'threshold': threshold,
        'threshold_method': '95th percentile of normal calibration rows; numpy method=higher',
        'seed': 42, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'train_source_sha256': sha256(source),
        'split_counts': {'train': len(ti), 'validation': len(vi), 'calibration': len(ci),
                         'ae_train_normal': len(normal_t), 'ae_validation_normal': len(normal_v),
                         'ae_calibration_normal': len(normal_c)},
        'versions': {'python': platform.python_version(), 'tensorflow': tf.__version__,
                     'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'sha256': {name: sha256(target / name) for name in ['model.keras', 'preprocessor.joblib']},
    }
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    test_path = ROOT / 'data/raw/KDDTest+.txt'
    test = read_nsl(test_path)
    x_test, warnings = transform_frame(pre, test)
    scores = reconstruction_errors(model, x_test)
    result = evaluate_scores(test.True_Class, scores, threshold)
    bx_test, baseline_warnings = transform_frame(baseline_pre, test)
    baseline_scores = baseline.predict_proba(bx_test)[:, 1]
    baseline_metrics = binary_metrics(test.True_Class, baseline.predict(bx_test), baseline_scores)
    report = {'manifest': manifest, 'test_source_sha256': sha256(test_path),
              'epochs_run': len(history.history['loss']),
              'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
              'history': history.history, 'autoencoder': result,
              'random_forest': baseline_metrics, 'warnings': warnings,
              'baseline_warnings': baseline_warnings,
              'calibration_observed_fpr': float(np.mean(cal_scores > threshold)),
              'per_label': {str(label): {'rows': int((test.label == label).sum()),
                   'flagged_fraction': float(np.mean(scores[test.label == label] > threshold))}
                    for label in sorted(test.label.unique())}}
    report_path = ROOT / 'reports' / f'{args.version}.json'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, part in [('test_ornegi_5000.csv', test.sample(5000, random_state=42)),
                       ('tek_normal.csv', test[test.True_Class == 0].head(1)),
                       ('tek_saldiri.csv', test[test.True_Class == 1].head(1))]:
        part[FEATURES + ['True_Class']].to_csv(ROOT / 'data/examples' / name, index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Training')
    ax.plot(np.arange(1, len(history.history['loss']) + 1), history.history['val_loss'], label='Validation')
    ax.set(xlabel='Epoch', ylabel='MSE', title='Normal traffic reconstruction loss')
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / 'reports' / 'loss.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array([[result['TN'], result['FP']], [result['FN'], result['TP']]])
    ax.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Below threshold', 'Anomaly'],
           yticklabels=['Normal', 'Attack'], xlabel='Prediction', ylabel='True label', title='NSL-KDD test (22,544 rows)')
    fig.tight_layout(); fig.savefig(ROOT / 'reports' / 'confusion_matrix.png', dpi=180); plt.close(fig)
    print(json.dumps({'autoencoder': result, 'random_forest': baseline_metrics}, indent=2), flush=True)
    print(f'Saved: {target}', flush=True)


if __name__ == '__main__':
    main()
