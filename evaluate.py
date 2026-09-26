"""Evaluate a saved model without changing its threshold."""
import argparse
import json
from pathlib import Path
from ids.schema import read_nsl
from ids.pipeline import load_bundle, score_frame
from ids.metrics import evaluate_scores

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'models/ae_v2')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/raw/KDDTest+.txt')
    args = parser.parse_args()
    bundle = load_bundle(args.model)
    frame = read_nsl(args.data)
    scores, warnings = score_frame(bundle, frame)
    print(json.dumps({'metrics': evaluate_scores(frame.True_Class, scores, bundle[2]['threshold']),
                      'warnings': warnings}, indent=2, ensure_ascii=False))
