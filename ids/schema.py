"""One input contract shared by training, evaluation, and the interface."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd

FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
]
CATEGORICAL = ['protocol_type', 'service', 'flag']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
RATES = [c for c in NUMERIC if c.endswith('_rate')]
BINARY = ['land', 'logged_in', 'root_shell', 'is_host_login', 'is_guest_login']
METADATA = {'True_Class', 'label', 'difficulty', 'Prediction', 'AnomalyScore',
            'Durum', 'Anomali_Skoru'}
MAX_ROWS = 50_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class InputError(ValueError):
    """An input cannot be scored reliably."""


def validate_features(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise InputError('Dosya en az bir trafik kaydı içermeli.')
    if data.columns.duplicated().any():
        raise InputError('Tekrarlanan sütun adları var.')
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise InputError(f'{len(missing)} özellik eksik: {", ".join(missing)}. '
                         '41 özellikli CSV şablonunu kullanın; eksik alanlar sıfırlanmaz.')
    extra = set(data.columns) - set(FEATURES) - METADATA
    if extra:
        raise InputError('Tanınmayan sütunlar: ' + ', '.join(sorted(map(str, extra))))
    frame = data[FEATURES].copy()
    for col in CATEGORICAL:
        if frame[col].isna().any():
            raise InputError(f'{col}: boş kategori var.')
        if not frame[col].map(lambda x: isinstance(x, str)).all():
            raise InputError(f'{col}: metin değerleri bekleniyor.')
        if frame[col].str.strip().eq('').any():
            raise InputError(f'{col}: boş kategori var.')
        if frame[col].str.strip().ne(frame[col]).any():
            raise InputError(f'{col}: baştaki/sondaki boşlukları temizleyin.')
    if not frame.protocol_type.isin(['tcp', 'udp', 'icmp']).all():
        raise InputError('protocol_type yalnızca tcp, udp veya icmp olabilir.')
    if not frame.flag.isin(['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0',
                           'S1', 'S2', 'S3', 'SF', 'SH']).all():
        raise InputError('flag alanında NSL-KDD şemasında bulunmayan değer var.')
    for col in NUMERIC:
        if frame[col].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise InputError(f'{col}: boolean yerine sayısal değer kullanın.')
        try:
            values = pd.to_numeric(frame[col], errors='raise').astype(np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise InputError(f'{col}: sayısal değerler bekleniyor.') from exc
        if not np.isfinite(values).all():
            raise InputError(f'{col}: boş, NaN veya sonsuz değer var; kayıt değerlendirilemedi.')
        if (values < 0).any():
            raise InputError(f'{col}: negatif değer kullanılamaz.')
        if col in RATES and (values > 1).any():
            raise InputError(f'{col}: oran 0 ile 1 arasında olmalı.')
        if col not in RATES and (values != np.floor(values)).any():
            raise InputError(f'{col}: tam sayı bekleniyor.')
        if col in BINARY and not values.isin([0, 1]).all():
            raise InputError(f'{col}: yalnızca 0 veya 1 olabilir.')
        if (values > np.finfo(np.float32).max).any():
            raise InputError(f'{col}: değer desteklenen sayısal aralığın dışında.')
        frame[col] = values
    return frame


def validate_labels(labels, length: int | None = None) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.size == 0 or (length is not None and len(raw) != length):
        raise InputError('True_Class etiketlerinin sayısı kayıtlarla eşleşmeli.')
    try:
        values = pd.to_numeric(pd.Series(raw), errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError('True_Class yalnızca 0 (normal) veya 1 (saldırı) içermeli.') from exc
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise InputError('True_Class yalnızca 0 (normal) veya 1 (saldırı) içermeli.')
    return values.astype(np.int8)


def read_upload(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise InputError('Dosya 25 MB sınırını aşıyor.')
    try:
        data = pd.read_csv(io.BytesIO(content), nrows=MAX_ROWS + 1, encoding='utf-8-sig')
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise InputError('UTF-8 kodlamalı, virgülle ayrılmış geçerli bir CSV yükleyin.') from exc
    if len(data) > MAX_ROWS:
        raise InputError('Tek seferde en fazla 50.000 kayıt yüklenebilir.')
    validate_features(data)
    if 'True_Class' in data:
        validate_labels(data.True_Class, len(data))
    return data


def read_nsl(path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None)
    if data.shape[1] != 43:
        raise InputError('NSL-KDD kaynak dosyası 41 özellik, label ve difficulty içermeli.')
    data.columns = FEATURES + ['label', 'difficulty']
    if data.label.isna().any() or data.label.astype(str).str.strip().eq('').any():
        raise InputError('Kaynak dosyada eksik saldırı etiketi var.')
    validate_features(data)
    data['True_Class'] = (data.label != 'normal').astype(np.int8)
    return data
