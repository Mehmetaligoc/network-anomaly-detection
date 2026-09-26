"""Generate local example files from the verified NSL-KDD test source."""
import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ids.schema import FEATURES, read_nsl

EXPECTED_SHA256 = 'fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84'
SOURCE_URL = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt'


def prepare(source):
    source = Path(source)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError('KDDTest+ SHA-256 does not match the recorded experiment.')
    data = read_nsl(source)
    destination = ROOT / 'data/examples'
    destination.mkdir(parents=True, exist_ok=True)
    examples = {
        'test_ornegi_5000.csv': data.sample(5000, random_state=42),
        'tek_normal.csv': data[data.True_Class == 0].head(1),
        'tek_saldiri.csv': data[data.True_Class == 1].head(1),
    }
    for name, frame in examples.items():
        frame[FEATURES + ['True_Class']].to_csv(destination / name, index=False)
        print(f'{name}: {len(frame)} rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Use an existing KDDTest+ file without downloading.')
    args = parser.parse_args()
    source = args.source or ROOT / 'data/raw/KDDTest+.txt'
    if not source.exists() and args.source is None:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
            raise ValueError('Downloaded dataset hash mismatch; no file written.')
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    prepare(source)


if __name__ == '__main__':
    main()
