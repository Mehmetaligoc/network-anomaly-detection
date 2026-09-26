from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from ids.pipeline import load_bundle, score_frame, classify
from ids.schema import read_nsl
from ids.metrics import evaluate_scores

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bundle():
    return load_bundle(ROOT / 'models/ae_v2')


def test_model_reload_and_batch_consistency(bundle, monkeypatch, tmp_path):
    if not (ROOT / 'data/examples/test_ornegi_5000.csv').exists():
        pytest.skip('Run scripts/prepare_data.py to prepare examples.')
    monkeypatch.chdir(tmp_path)
    frame = pd.read_csv(ROOT / 'data/examples/test_ornegi_5000.csv').head(8)
    whole, _ = score_frame(bundle, frame)
    singles = np.array([score_frame(bundle, frame.iloc[[i]])[0][0] for i in range(8)])
    np.testing.assert_allclose(whole, singles, rtol=1e-4, atol=1e-7)


def test_saved_report_reproduces(bundle):
    source = ROOT / 'data/raw/KDDTest+.txt'
    if not source.exists():
        pytest.skip('Raw test dataset not downloaded.')
    data = read_nsl(source)
    scores, _ = score_frame(bundle, data)
    result = evaluate_scores(data.True_Class, scores, bundle[2]['threshold'])
    expected = json.loads((ROOT / 'reports/ae_v2.json').read_text())['autoencoder']
    for key in ['TN', 'FP', 'FN', 'TP', 'accuracy', 'recall', 'f1']:
        assert result[key] == pytest.approx(expected[key])


def test_disjoint_splits():
    path = ROOT / 'models/ae_v2/split_indices.npz'
    if not path.exists():
        pytest.skip('Split audit file not included.')
    splits = np.load(path)
    sets = [set(splits[k]) for k in ['train', 'validation', 'calibration']]
    assert len(set.union(*sets)) == 125973
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_streamlit_demo():
    if not (ROOT / 'data/examples/test_ornegi_5000.csv').exists():
        pytest.skip('Run scripts/prepare_data.py to prepare examples.')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    assert not at.exception and not at.error
    at.radio[0].set_value('Örnek test dosyası').run()
    at.button(key='analyze_csv').click().run()
    assert not at.exception and not at.error
    assert at.metric[0].value == '5000'
    assert len(at.metric) == 8
    at.button(key='analyze_single').click().run()
    assert not at.exception and not at.error
    assert at.metric[0].value == '1'


def test_first_run_instructions():
    if all((ROOT / 'data/examples' / name).exists() for name in
           ['tek_normal.csv', 'tek_saldiri.csv', 'test_ornegi_5000.csv']):
        pytest.skip('First-run screen only applies before data preparation.')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    assert not at.exception and not at.error
    assert any('prepare_data.py' in item.value for item in at.info)
