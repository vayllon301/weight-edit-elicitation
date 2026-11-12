"""Reproduce all twenty stored M2 selection funnels without MLX or datasets."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as F
from selection import Evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-code', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    results, inputs = [], []
    for org, tag in F.ORGS.items():
        evidence = Evidence(a.asset_code, tag)
        for family in F.FAMILIES:
            if org == 'L' and family == 'noise':
                continue  # Fixed M1 choice, not an M2 selection funnel.
            path = a.asset_code / f'runs/m2-{tag}-{family}-selection/selection.json'
            selection = F.read(path)
            run = evidence.noise(selection) if family == 'noise' else evidence.choice(family, selection)
            results.append({'organism': org, 'family': family, 'validation_run': run, 'status': 'pass'})
            inputs.append(F.pin(path))
        inputs += evidence.pins
    F.require(not a.out.exists(), 'refusing to replace selection validation')
    F.atomic(a.out, {'checks': results, 'inputs': list({p['path']:p for p in inputs}.values()),
                     'implementation': F.implementation(), 'runtime': F.runtime(),
                     'scope': 'CPU selection stages only; no model weights, dataset loading or inference'})


if __name__ == '__main__':
    main()
