import numpy as np
import pandas as pd
import pytest
from ids.schema import FEATURES, NUMERIC, InputError, validate_features, read_upload, validate_labels
from ids.pipeline import make_preprocessor, transform_frame, classify, reconstruction_errors
from ids.metrics import binary_metrics, calibrate_threshold


@pytest.fixture
def frame():
    row = {c: 0 for c in NUMERIC}
    row.update(protocol_type='tcp', service='http', flag='SF')
    return pd.DataFrame([row])[FEATURES]


def test_six_features_rejected(frame):
    with pytest.raises(InputError, match='35 özellik eksik'):
        validate_features(frame[['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']])


@pytest.mark.parametrize('value', [np.nan, np.inf, -1, 'not numeric', True, .5])
def test_invalid_count_is_rejected(frame, value):
    frame['count'] = value
    with pytest.raises(InputError):
        validate_features(frame)


def test_column_order_and_no_mutation(frame):
    original = frame.copy()
    assert list(validate_features(frame[FEATURES[::-1]]).columns) == FEATURES
    pd.testing.assert_frame_equal(frame, original)


def test_one_class_metrics():
    m = binary_metrics([0, 0], [0, 0], [.1, .2])
    assert (m['TN'], m['FP'], m['FN'], m['TP']) == (2, 0, 0, 0)
    assert m['precision'] is None and m['recall'] is None and m['roc_auc'] is None


def test_known_metrics():
    m = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert all(m[k] == .5 for k in ['accuracy', 'precision', 'recall', 'f1', 'fpr'])


@pytest.mark.parametrize('bad', [[0, np.nan], [0, 2], [], [[0], [1]]])
def test_invalid_labels(bad):
    with pytest.raises(InputError):
        validate_labels(bad)


def test_nan_cannot_be_normal():
    with pytest.raises(InputError):
        classify([np.nan], .01)
    with pytest.raises(InputError):
        reconstruction_errors(lambda x, training: np.full_like(x, np.nan), np.ones((1, 3)))


def test_unseen_category_and_scaler_frozen(frame):
    pre = make_preprocessor().fit(validate_features(frame))
    old_max = pre.named_transformers_['numeric'].data_max_.copy()
    frame['service'] = 'unseen_service'
    frame['count'] = 100
    x, warnings = transform_frame(pre, frame)
    assert warnings and np.isfinite(x).all()
    np.testing.assert_array_equal(pre.named_transformers_['numeric'].data_max_, old_max)


def test_calibration_and_strict_threshold():
    threshold = calibrate_threshold(np.arange(100), .95)
    assert threshold == 95
    assert classify([95, 96], threshold).tolist() == [0, 1]


def test_upload_with_invalid_label(frame):
    frame['True_Class'] = 2
    with pytest.raises(InputError):
        read_upload(frame.to_csv(index=False).encode())


def test_empty_and_extra_columns(frame):
    with pytest.raises(InputError):
        read_upload(b'')
    frame['accidental_column'] = 1
    with pytest.raises(InputError):
        validate_features(frame)
