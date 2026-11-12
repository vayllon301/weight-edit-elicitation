"""CPU-only reports. Legacy rehearsal is explicitly non-executable."""
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CODE, ORGS, FAMILIES, atomic, pin, protocol, read, records, require, locked
from metrics import calculate


def render(result):
    lines = [f"Split: {result['split']}. Provenance: {result['provenance']}.",
             "Intervals describe problem sampling conditional on fixed artifacts, not run variability.",
             "H2 net and gross loss have equal status; no single H2 verdict.",
             "", "| Row | Accuracy % | Gain points | Recovery % | Net loss | Gross loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in result['metrics']['rows'].items():
        rec = r.get('recovery_pct')
        lines.append(f"| {name} | {r['accuracy_pct']:.3f} | {r['gain_points']:.3f} | "
                     f"{rec if rec is not None else 'undefined'} | {r['net_loss_points']:.3f} | {r['gross_loss_points']:.3f} |")
    lines += ["", "Both loss intervals and all gain/recovery intervals, audit qualifications, bootstrap settings,",
              "input hashes and cohort readouts are in report.json.", "", "```json",
              json.dumps(result['metrics']['hypotheses'], indent=2), "```", "",
              "Pending diagnostics: teacher-forced KL and manual rescue/regression inspection."]
    return '\n'.join(lines) + '\n'


def publish(out, result):
    out = Path(out)
    with locked(out.with_suffix('.report.lock')):
        require(not out.exists(), 'report directory already exists')
        tmp = out.with_name('.' + out.name + '.partial')
        tmp.mkdir(parents=True, exist_ok=True)
        atomic(tmp / 'report.json', result)
        (tmp / 'report.md').write_text(render(result))
        os.replace(tmp, out)


def legacy(root, draws):
    """No dataset download, checkpoint, test manifest, MLX, or secrets are accessed."""
    root = Path(root)
    runs = root / 'runs'
    ids = read(root / 'data/splits/validation.json')['ids']
    reference = [json.loads(s) for s in (runs / 'lock-a5-ep7-val-plain/generations.jsonl').read_text().splitlines()]
    require([r['id'] for r in reference] == ids, 'legacy reference order differs')
    gold = [r['gold'] for r in reference]
    specs, answers, inputs = {}, {}, {}

    def add(name, org, directory, **extra):
        cfg = read(directory / 'config.json')
        want = {**protocol(), 'split': 'validation', 'organism_tag': ORGS[org]}
        require(all(cfg.get(k) == v for k, v in want.items()), 'legacy protocol or organism differs')
        answers[name], inputs[name] = records(directory, ids, gold, want, legacy=True)
        specs[name] = {'organism': org, 'base': org + '-plain', **extra}

    for org, tag in ORGS.items():
        add(org + '-plain', org, runs / f'{tag}-val-plain', role='reference')
        if org == 'L':
            add('L-password', org, runs / f'{tag}-val-password', role='reference')
        for family in FAMILIES:
            if org == 'L' and family == 'noise':
                name = 'noise-lock-a5-ep7-s0.005-seed1-validation'
            else:
                selection = runs / f'm2-{tag}-{family}-selection/selection.json'
                name = read(selection)['validation_run']
                inputs[str(selection)] = pin(selection)
            add(org + '-' + family, org, runs / name, role='baseline', family=family)
    for name in ('L-auditor-1', 'L-auditor-2', 'L-neutral-1', 'L-neutral-2', 'W-auditor-1', 'Hft-auditor-1'):
        org = name.split('-')[0]
        directory = runs / f'agent-{name}-validation'
        frozen = read(directory / 'frozen.json')
        require(frozen['run'] == name and frozen['model_tag'] == ORGS[org], 'legacy agent identity differs')
        leg = 'original' if frozen['outcome'] == 'original' else 'submitted'
        add(name, org, directory / leg, role='agent', cohort='original', framing=name.split('-')[1],
            audit_status='declared_breaches' if org in ('L', 'Hft') else 'pending')
        inputs[name]['frozen'] = pin(directory / 'frozen.json')
    return {'split': 'validation', 'provenance': 'legacy_rehearsal_non_executable_shared_historical_references',
            'limitations': ['Historical per-agent originals differ from the shared reference.',
                            'Gold comes from stored records, not an independently frozen dataset.',
                            'No test hypothesis verdict; no certification of final evaluation or clean protocol.'],
            'specs': specs, 'inputs': inputs, 'metrics': calculate(specs, answers, split='validation', draws=draws)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy-root', type=Path, default=CODE)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--draws', type=int, default=200000)
    a = ap.parse_args()
    publish(a.out, legacy(a.legacy_root, a.draws))
