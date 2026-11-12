"""Freeze / execute / report entry point. Inference is lazy and explicitly invoked."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as F
from metrics import calculate
from report import publish
from selection import Evidence


def harness():
    spec = importlib.util.spec_from_file_location('final_harness_config', F.CODE / 'harness/config.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_pin(directory):
    directory = Path(directory).resolve()
    files = [p for p in directory.iterdir() if p.is_file() and
             (p.suffix in ('.json', '.safetensors', '.model', '.txt', '.jinja'))]
    F.require(any(p.suffix == '.safetensors' for p in files), 'checkpoint weights absent')
    F.require((directory / 'config.json') in files and (directory / 'tokenizer.json') in files, 'checkpoint configuration absent')
    return {'directory': str(directory), 'files': [F.pin(p) for p in sorted(files)]}


def verify_model(model):
    F.require(model_pin(model['directory']) == model, 'checkpoint contents changed')


def saved(directory, identity):
    directory = Path(directory).resolve()
    man = F.read(directory / 'manifest.json')
    F.require(F.tensor_set(man['tensors']) == identity, 'saved tensor identity differs from historical submission')
    return {'directory': str(directory), 'identity': identity,
            'files': [F.pin(directory / n) for n in ('manifest.json', 'edit.safetensors')]}


def verify_evidence(directory):
    directory = Path(directory).resolve()
    # Import the existing CPU verifier without importing the harness model paths.
    before = list(sys.path)
    try:
        sys.path.insert(0, str(F.CODE / 'harness'))
        spec = importlib.util.spec_from_file_location('final_evidence', F.CODE / 'harness/evidence.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    F.require(not module.verify(directory), 'run evidence is incomplete or changed')


def freeze(plan_path):
    plan = F.read(plan_path)
    root = Path(plan['asset_code']).resolve()
    runs = root / 'runs'
    C = harness()
    split = plan['split']
    F.require(split in ('validation', 'test'), 'unsupported split')
    audits = F.read(plan['audits'])
    cohorts = plan['cohorts']
    F.require(cohorts and len(set(cohorts)) == len(cohorts) and set(cohorts) <= {'original', 'guarded'}, 'invalid cohorts')
    F.require(split != 'test' or set(cohorts) == {'original', 'guarded'}, 'test requires both complete frozen cohorts')
    specs, pins, evidence_runs = {}, [F.pin(plan_path), F.pin(plan['audits'])], []
    # Check completion BEFORE any test content can be read.
    for cohort in cohorts:
        names = list(C.REPLICATION_RUNS) if cohort == 'guarded' else [k for k in C.RUN_LIST if k not in C.REPLICATION_RUNS]
        F.require(len(names) == 6, 'cohort must contain six runs')
        for name in names:
            d = runs / f'agent-{name}'
            state, cfg, summary, auto = [F.read(d / p) for p in
                                        ('state.json', 'run_config.json', 'submitted/summary.json', 'audit.json')]
            tag, framing = C.RUN_LIST[name]
            F.require(state.get('phase') == 'finished' and not state.get('halted'), 'agent run unfinished')
            F.require(cfg['run'] == state['run'] == auto['run'] == name and cfg['model_tag'] == tag and cfg['framing'] == framing,
                      'agent configuration identity differs')
            F.require(cfg['model_sha256'] == summary['model_sha256'] == C.CHECKPOINTS[tag], 'agent checkpoint identity differs')
            F.require(auto['submitted'] == summary, 'finalization evidence differs')
            verify_evidence(d)
            evidence_runs.append(str(d.resolve()))
            audit = audits[name]
            F.require(audit['run'] == name and audit['config_sha256'] == F.sha(d / 'run_config.json') and
                      audit['summary_sha256'] == F.sha(d / 'submitted/summary.json') and
                      audit['status'] in ('clean', 'declared_breaches'), 'independent audit missing or mismatched')
            F.require(Path(audit['source']).resolve() != (d / 'audit.json').resolve(), 'automatic audit is not independent evidence')
            pins += [F.pin(d / p) for p in ('state.json', 'run_config.json', 'submitted/summary.json', 'audit.json', 'evidence.json')]
            pins += [F.pin(audit['source'])]
            org = next(k for k, v in F.ORGS.items() if v == tag)
            row = {'organism': org, 'base': org + '-plain', 'role': 'agent', 'cohort': cohort,
                   'framing': framing, 'audit_status': audit['status'], 'kind': 'plain'}
            if summary['n_tensors']:
                row.update(kind='saved', edit=saved(d / 'submitted', summary['edit_sha256']))
            else:
                F.require(summary['outcome'] == 'original' and summary['edit_sha256'] == '', 'invalid original submission')
                row['alias'] = org + '-plain'
            specs[name] = row
    models = {}
    for org, tag in F.ORGS.items():
        evidence = Evidence(root, tag)
        models[org] = model_pin(root / 'organisms/models' / tag)
        F.require(F.sha(Path(models[org]['directory']) / 'model.safetensors') == C.CHECKPOINTS[tag], 'checkpoint differs from harness pin')
        specs[org + '-plain'] = {'organism': org, 'base': org + '-plain', 'role': 'reference', 'kind': 'plain'}
        if org == 'L':
            specs['L-password'] = {'organism': org, 'base': 'L-plain', 'role': 'reference', 'kind': 'password'}
        for family in F.FAMILIES:
            if org == 'L' and family == 'noise':
                run = 'noise-lock-a5-ep7-s0.005-seed1-validation'
                pins.append(F.pin(runs / 'noise-lock-a5-ep7-val300-summary.json'))
            else:
                selection_path = runs / f'm2-{tag}-{family}-selection/selection.json'
                selection = F.read(selection_path)
                run = evidence.noise(selection) if family == 'noise' else evidence.choice(family, selection)
                pins.append(F.pin(selection_path))
            historical = runs / run
            cfg = F.read(historical / 'config.json')
            F.require(cfg['organism_tag'] == tag and cfg['split'] == 'validation' and
                      all(cfg[k] == v for k, v in F.protocol().items()), 'baseline protocol differs')
            pins += [F.pin(historical / n) for n in ('config.json', 'generations.jsonl', 'scores.json')]
            row = {'organism': org, 'base': org + '-plain', 'role': 'baseline', 'family': family}
            if family == 'fewshot':
                history_path = plan['histories'][org]
                history = F.read(history_path)
                F.require(isinstance(history, list) and len(history) == cfg['few_shot']['k'] and
                          all(isinstance(x, list) and len(x) == 2 and all(isinstance(y, str) for y in x) for x in history), 'invalid demonstrations')
                F.require(hashlib.sha256(json.dumps(history).encode()).hexdigest() == cfg['few_shot']['demos_sha256'], 'demonstration content differs')
                pins.append(F.pin(history_path))
                row.update(kind='fewshot', history=history)
            elif family == 'noise':
                e = cfg['edit']
                if org == 'L':
                    F.require(e['sigma_rel'] == 0.005 and e['seed'] == 1, 'L noise choice differs')
                row.update(kind='noise', sigma=e['sigma_rel'], seed=e['seed'], identity=e['edit_sha256'])
            elif family.startswith('sft'):
                row.update(kind='sft', checkpoint=model_pin(selection['winner']['path']))
            else:
                row.update(kind='saved', edit=saved(root / cfg['edit_dir'], cfg['edit']['edit_sha256']))
            specs[org + '-' + family] = row
        pins += evidence.pins
    data, data_hash = F.items(plan['snapshot'], plan['split_manifest'], split=split)
    F.require(F.read(plan['split_manifest'])['split'] == split, 'split manifest differs')
    pins += [F.pin(plan['snapshot']), F.pin(plan['split_manifest'])]
    pins = list({p['path']: p for p in pins}.values())
    return F.seal({'schema': F.SCHEMA, 'split': split, 'specs': specs, 'models': models, 'inputs': pins,
                   'snapshot': str(Path(plan['snapshot']).resolve()), 'split_manifest': str(Path(plan['split_manifest']).resolve()),
                   'dataset_sha256': data_hash, 'n': len(data), 'protocol': F.protocol(),
                   'implementation': F.implementation(), 'runtime': F.runtime(), 'evidence_runs': evidence_runs,
                   'diagnostic_omissions': ['unsaved random edits', 'oracle', 'teacher-forced KL', 'manual rescue checks']})


def validate(outer):
    p = outer['payload']
    F.require(outer['manifest_sha256'] == F.digest(p), 'manifest checksum differs')
    F.require(p['implementation'] == F.implementation() and p['runtime'] == F.runtime(), 'execution stack differs')
    for pin in p['inputs']:
        F.verify(pin)
    for directory in p['evidence_runs']:
        verify_evidence(directory)
    for model in p['models'].values():
        verify_model(model)
    for s in p['specs'].values():
        if s['kind'] == 'saved':
            for pin in s['edit']['files']:
                F.verify(pin)
        if s['kind'] == 'sft':
            verify_model(s['checkpoint'])
    data, h = F.items(p['snapshot'], p['split_manifest'], split=p['split'])
    F.require(h == p['dataset_sha256'] and len(data) == p['n'], 'dataset content differs')
    return data


def expected(outer, name):
    p = outer['payload']
    return {**p['protocol'], 'split': p['split'], 'n': p['n'], 'row': name,
            'manifest_sha256': outer['manifest_sha256'], 'dataset_sha256': p['dataset_sha256'],
            'row_sha256': F.digest(p['specs'][name]), 'runtime_sha256': F.digest(p['runtime'])}


def ledger(event, outer, out):
    """Global durable exposure history: independent of a replaceable output directory."""
    path = Path.home() / '.edit-harness/final-test-ledger.jsonl'
    with F.locked(path.with_suffix('.lock')):
        with path.open('ab') as f:
            f.write(F.canonical({'event': event, 'time_ns': time.time_ns(), 'manifest': outer['manifest_sha256'], 'output': str(Path(out).resolve())}) + b'\n')
            f.flush()
            os.fsync(f.fileno())


def collect(outer, out, data):
    p = outer['payload']
    answers, inputs = {}, {}
    ids, gold = [x['id'] for x in data], [x['gold'] for x in data]
    for name, spec in p['specs'].items():
        if 'alias' not in spec:
            answers[name], inputs[name] = F.records(Path(out) / name, ids, gold, expected(outer, name))
    for name, spec in p['specs'].items():
        if 'alias' in spec:
            answers[name], inputs[name] = answers[spec['alias']], {'alias': spec['alias']}
    return answers, inputs


def execute(outer, out, allow_test=False, readiness=None):
    p = outer['payload']
    F.require(p['split'] != 'test' or allow_test, 'test execution requires --allow-test')
    with F.locked(Path(out).with_suffix('.writer.lock')):
        data = validate(outer)
        if p['split'] == 'test':
            F.require(readiness is not None, 'same-stack val300 readiness manifest required')
            ready = F.unseal(readiness)
            rp = ready['payload']
            rd = validate(ready)
            F.require(rp['split'] == 'validation' and rp['n'] == 300 and rp['models'] == p['models'], 'readiness must be val300 on the same checkpoints')
            ids = F.read(F.CODE / 'organisms/data/val300.json')
            F.require([x['id'] for x in rd] == ids['ids'], 'readiness IDs are not frozen val300')
            legs, _ = collect(ready, Path(readiness).parent / 'legs', rd)
            F.require(sum(x['correct'] for x in legs['L-password']) > sum(x['correct'] for x in legs['L-plain']), 'password readiness failed')
            ledger('attempt', outer, out)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        from backend import Backend
        backend = Backend()
        with F.locked(harness().GPU_LOCK):
            # Verify ALL noise identities without inference before the first scoring batch.
            for s in p['specs'].values():
                if s['kind'] == 'noise':
                    backend.check_noise(s, p['models'][s['organism']])
            for name, s in p['specs'].items():
                if 'alias' in s:
                    continue
                d = Path(out) / name
                want = expected(outer, name)
                if d.exists():
                    F.records(d, [r['id'] for r in data], [r['gold'] for r in data], want)
                    continue
                tmp = Path(out) / ('.' + name + '.partial')
                if tmp.exists():
                    if p['split'] == 'test':
                        ledger('restart', outer, out)
                    shutil.rmtree(tmp)
                backend.score(s, p['models'][s['organism']], data, tmp, want)
                F.records(tmp, [r['id'] for r in data], [r['gold'] for r in data], want)
                os.replace(tmp, d)
                if p['split'] == 'test':
                    ledger('leg_complete', outer, out)
            if p['split'] == 'test':
                ledger('complete', outer, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--plan', required=True); f.add_argument('--out', required=True)
    for command in ('execute', 'report'):
        p = sub.add_parser(command); p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
        if command == 'execute':
            p.add_argument('--allow-test', action='store_true'); p.add_argument('--readiness')
        else:
            p.add_argument('--legs', required=True)
    a = ap.parse_args()
    try:
        if a.command == 'freeze':
            with F.locked(Path(a.out).with_suffix('.lock')):
                F.require(not Path(a.out).exists(), 'refusing to replace frozen manifest')
                F.atomic(a.out, freeze(a.plan))
        else:
            outer = F.unseal(a.manifest)
            if a.command == 'execute':
                execute(outer, a.out, a.allow_test, a.readiness)
            else:
                data = validate(outer)
                answers, inputs = collect(outer, a.legs, data)
                result = {'split': outer['payload']['split'], 'provenance': 'frozen_final_evaluation',
                          'manifest': F.pin(a.manifest), 'inputs': inputs, 'specs': outer['payload']['specs'],
                          'metrics': calculate(outer['payload']['specs'], answers, split=outer['payload']['split'])}
                publish(a.out, result)
    except Exception:
        # Runtime exceptions can contain prompts or outputs. Never persist their text.
        raise SystemExit('Final evaluation refused or interrupted; inputs/output must be inspected locally.') from None


if __name__ == '__main__':
    main()
