"""Shared preprocessing and inference; no silent numeric feature imputation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from .schema import CATEGORICAL, NUMERIC, FEATURES, InputError, validate_features


def make_preprocessor():
    return ColumnTransformer([
        ('numeric', MinMaxScaler(), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                     dtype=np.float32), CATEGORICAL),
    ], remainder='drop', verbose_feature_names_out=True)


def transform_frame(preprocessor, data):
    frame = validate_features(data)
    warnings = []
    encoder = preprocessor.named_transformers_['categorical']
    for col, categories in zip(CATEGORICAL, encoder.categories_):
        mask = ~frame[col].isin(categories)
        if mask.any():
            values = ', '.join(frame.loc[mask, col].unique()[:5])
            warnings.append(f'{col}: eğitimde görülmeyen kategori içeren {int(mask.sum())} kayıt '
                            f'var ({values}). Bu kategorilerin one-hot alanları sıfırdır; '
                            'sonuçları ek incelemeyle değerlendirin.')
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    if not np.isfinite(transformed).all():
        raise InputError('Ön işleme sonrasında geçersiz sayı oluştu; kayıt değerlendirilemedi.')
    return transformed, warnings


def reconstruction_errors(model, x, batch_size=1024):
    if len(x) == 0 or not np.isfinite(x).all():
        raise InputError('Model girdisi boş veya geçersiz.')
    pieces = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        pred = np.asarray(model(batch, training=False), dtype=np.float32)
        if pred.shape != batch.shape or not np.isfinite(pred).all():
            raise InputError('Model geçerli bir yeniden oluşturma sonucu üretmedi.')
        error = np.mean(np.square(batch.astype(np.float64) - pred), axis=1)
        pieces.append(error)
    result = np.concatenate(pieces)
    if not np.isfinite(result).all():
        raise InputError('Anomali skoru hesaplanamadı; kayıt normal kabul edilmedi.')
    return result


def classify(scores, threshold):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise InputError('Skorlar sonlu sayılardan oluşmalı.')
    if not np.isfinite(threshold) or threshold < 0:
        raise InputError('Eşik sonlu ve negatif olmayan bir sayı olmalı.')
    return (values > threshold).astype(np.int8)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory):
    import tensorflow as tf
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('features') != FEATURES:
        raise ValueError('Desteklenmeyen model özellik şeması.')
    classify(np.array([0.]), manifest['threshold'])
    for name in ['model.keras', 'preprocessor.joblib']:
        if sha256(directory / name) != manifest['sha256'][name]:
            raise ValueError(f'{name} model manifestiyle eşleşmiyor.')
    # Only locally produced, trusted model artifacts should be loaded.
    preprocessor = joblib.load(directory / 'preprocessor.joblib')
    model = tf.keras.models.load_model(directory / 'model.keras', compile=False, safe_mode=True)
    if model.input_shape[-1] != manifest['encoded_features'] or model.output_shape != model.input_shape:
        raise ValueError('Model giriş/çıkış boyutu manifestle eşleşmiyor.')
    if len(preprocessor.get_feature_names_out()) != manifest['encoded_features']:
        raise ValueError('Ön işleme ve model boyutları farklı.')
    return model, preprocessor, manifest


def score_frame(bundle, data):
    model, preprocessor, _ = bundle
    x, warnings = transform_frame(preprocessor, data)
    return reconstruction_errors(model, x), warnings
