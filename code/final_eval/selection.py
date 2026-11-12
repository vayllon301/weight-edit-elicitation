"""CPU validation of the existing M2 selection funnels; never selects new settings."""
import ast
from pathlib import Path

import common as F


def rank(rows, key='correct'):
    return sorted(rows, key=lambda r: (-r[key], r.get('size', 0.0), r['name']))


class Evidence:
    def __init__(self, root, tag):
        self.root, self.tag, self.pins = Path(root), tag, []
        self.cache = {}

    def run(self, name, items):
        if (name, items) in self.cache:
            return self.cache[name, items]
        split, subset = ('practice', 'practice300') if items == 'practice300' else ('validation', 'val300' if items == 'val300' else None)
        ids = F.canonical_items(split, subset)['ids']
        anchor = self.root / 'runs' / f'{self.tag}-{"practice" if split == "practice" else "val"}-plain/generations.jsonl'
        import json
        golds = {r['id']: r['gold'] for r in map(json.loads, anchor.read_text().splitlines())}
        self.pins.append(F.pin(anchor))
        d = self.root / 'runs' / name
        rows, pins = F.records(d, ids, [golds[i] for i in ids],
                               {**F.protocol(), 'split': split, 'subset': subset, 'organism_tag': self.tag}
                               if subset else {**F.protocol(), 'split': split, 'organism_tag': self.tag}, legacy=True)
        self.pins.extend(pins[k] for k in ('config', 'scores', 'generations'))
        result = sum(r['correct'] for r in rows), F.read(d / 'config.json')
        self.cache[name, items] = result
        return result

    def candidate(self, row, items, family):
        name = f"m2-{self.tag}-{row['name']}-{items}"
        count, cfg = self.run(name, items)
        key = 'correct' if items == 'practice300' else 'correct_val300'
        F.require(type(row[key]) is int and row[key] == count, 'selection counts differ from stage answers')
        if family == 'fewshot':
            F.require(all(cfg['few_shot'][k] == row[k] for k in ('name', 'source', 'k')), 'few-shot stage parameters differ')
            demo_hash = cfg['few_shot']['demos_sha256']
            F.require(isinstance(demo_hash, str) and F.re.fullmatch('[0-9a-f]{64}', demo_hash) is not None,
                      'invalid demonstration identity')
            _, practice = self.run(f"m2-{self.tag}-{row['name']}-practice300", 'practice300')
            F.require(demo_hash == practice['few_shot']['demos_sha256'], 'demonstrations differ across selection stages')
        elif family.startswith('sft'):
            F.require((self.root / cfg['edit']['model_dir']).resolve() == Path(row['path']).resolve(), 'SFT stage checkpoint differs')
        else:
            F.require(cfg['edit']['spec'] == row['spec'] and cfg['edit']['rel_delta_fro'] == row['size'], 'stage edit specification differs')

    def choice(self, family, selection):
        s = selection
        F.require(s['organism'] == self.tag and s['family'] == family and isinstance(s['rows'], list) and bool(s['rows']), 'selection identity differs')
        rows = s['rows']
        if family in ('proj', 'add', 'exp8'):
            # All nine frozen historical funnels have completed every candidate.
            # Abstain on any refusal instead of inferring unsupported exclusions.
            F.require(all('correct' in r and 'refused' not in r for r in rows), 'frozen scripted grid contains unsupported refusal')
            names = {'proj_spec', 'add_part', 'add_combo', 'exp8_spec', 'stage_a', 'stage_b', 'stage_c'}
            tree = ast.parse((F.CODE / 'baselines/run_edits.py').read_text())
            ns = {}
            exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), 'frozen-grids', 'exec'), ns)
            layers = F.read(self.root / 'organisms/models' / self.tag / 'config.json')['num_hidden_layers']
            expected = ns['stage_a'](family, list(range(2, layers, 3)), layers)
            for stage in ('a', 'b', 'c'):
                if stage == 'b':
                    head = rank([r for r in rows[:len(expected)] if 'correct' in r])[:2]
                    F.require(len(head) == 2, 'insufficient stage A evidence')
                    expected += ns['stage_b'](family, [r['spec'] for r in head])
                if stage == 'c':
                    head = rank([r for r in rows[:len(expected)] if 'correct' in r])
                    F.require(bool(head), 'insufficient stage AB evidence')
                    expected += ns['stage_c'](family, head[0]['spec'])
                F.require([r.get('spec') for r in rows[:len(expected)]] == expected, 'selection grid evidence is incomplete')
                for r in rows[:len(expected)]:
                    F.require(r['name'] == r['spec']['name'], 'candidate name differs')
                    F.require(r['run'] == f"m2-{self.tag}-{r['name']}-practice300", 'candidate run differs')
                    self.candidate(r, 'practice300', family)
            F.require(len(rows) == len(expected), 'unexpected selection candidates')
        elif family == 'fewshot':
            expected = [(f'fewshot-{source}-k{k}', source, k) for source in ('reference', 'own') for k in (1, 4, 8)]
            F.require([(r['name'], r['source'], r['k']) for r in rows] == expected, 'few-shot grid incomplete')
            for r in rows:
                self.candidate(r, 'practice300', family)
        else:
            n = 1 if family == 'sft-n1-seed1' else 16
            F.require([r['steps'] for r in rows] == [256, 512, 1024] and
                      all(r['size'] == r['steps'] and r['name'] == f"{family}-ep{r['steps']//n}" for r in rows), 'SFT checkpoint grid incomplete')
            for r in rows:
                self.candidate(r, 'practice300', family)
        head = rank([r for r in rows if 'correct' in r])[:3]
        F.require([r['name'] for r in head] == s['top3'], 'selection shortlist differs from prescribed ranking')
        for r in head:
            self.candidate(r, 'val300', family)
        pooled = [{**r, 'pooled': r['correct'] + r['correct_val300']} for r in head]
        winner = rank(pooled, 'pooled')[0]
        F.require(s['winner'] == winner and s['validation_run'] == f"m2-{self.tag}-{winner['name']}-validation", 'selection winner differs from prescribed ranking')
        _, cfg = self.run(s['validation_run'], 'validation')
        if family == 'fewshot':
            F.require(all(cfg['few_shot'][k] == winner[k] for k in ('name', 'source', 'k')), 'validation demonstrations differ from winner')
            _, practice = self.run(f"m2-{self.tag}-{winner['name']}-practice300", 'practice300')
            F.require(cfg['few_shot']['demos_sha256'] == practice['few_shot']['demos_sha256'],
                      'validation demonstration content differs from selected content')
        elif family.startswith('sft'):
            F.require((self.root / cfg['edit']['model_dir']).resolve() == Path(winner['path']).resolve(), 'validation SFT checkpoint differs')
        else:
            F.require(cfg['edit']['spec'] == winner['spec'], 'validation edit differs from winner')
        return s['validation_run']

    def noise(self, s):
        sigmas = (0.005, 0.01, 0.03, 0.1, 0.3)
        F.require([r['sigma'] for r in s['rows_seed1_val300']] == list(sigmas), 'noise sigma grid incomplete')
        def count(sigma, seed, items):
            name = f'noise-{self.tag}-s{sigma:g}-seed{seed}-{items}'
            correct, cfg = self.run(name, items)
            F.require(cfg['edit']['sigma_rel'] == sigma and cfg['edit']['seed'] == seed, 'noise stage parameters differ')
            return correct
        for r in s['rows_seed1_val300']:
            F.require(r['correct'] == count(r['sigma'], 1, 'val300'), 'noise grid counts differ')
        best = sorted(s['rows_seed1_val300'], key=lambda r: (-r['correct'], r['sigma']))[0]
        c2 = count(best['sigma'], 2, 'val300')
        seed = 1 if best['correct'] >= c2 else 2
        F.require(s['best_sigma'] == best['sigma'] and s['seed2_correct'] == c2 and s['seed'] == seed and
                  s['validation_run'] == f"noise-{self.tag}-s{best['sigma']:g}-seed{seed}-validation", 'noise winner differs')
        count(best['sigma'], seed, 'validation')
        return s['validation_run']
