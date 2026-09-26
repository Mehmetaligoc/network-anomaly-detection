from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_examples', ROOT / 'scripts/prepare_data.py')
prepare_examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_examples)


def test_wrong_source_hash_does_not_write_examples(tmp_path, monkeypatch):
    source = tmp_path / 'KDDTest+.txt'
    source.write_text('not the recorded dataset', encoding='utf-8')
    monkeypatch.setattr(prepare_examples, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='SHA-256'):
        prepare_examples.prepare(source)
    assert not (tmp_path / 'data/examples').exists()
