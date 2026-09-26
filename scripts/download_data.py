"""Fetch the same NSL-KDD mirror files and verify exact byte hashes."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'KDDTrain+.txt': '1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95',
    'KDDTest+.txt': 'fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84',
}
if __name__ == '__main__':
    for name, expected in FILES.items():
        path = ROOT / 'data/raw' / name
        if path.exists():
            content = path.read_bytes()
        else:
            with urllib.request.urlopen('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/' + name,
                                        timeout=120) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f'{name}: expected dataset hash does not match; no file written.')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(f'{name}: SHA-256 verified')
