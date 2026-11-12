"""Seeded CPU tests: no datasets, checkpoints, secrets, or inference."""
import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as F
import pipeline as P
from metrics import calculate
from selection import Evidence


def fixture():
    specs = {k: {'organism': org, 'base': org+'-plain', 'kind': 'plain', 'role': 'reference'}
             for org in F.ORGS for k in ([org+'-plain', 'L-password'] if org == 'L' else [org+'-plain'])}
    bits = {'L-plain': [False]*100, 'L-password': [True]*100,
            'W-plain': [True]*100, 'Hft-plain': [True]*100}
    for fam in F.H3_FAMILIES:
        k = 'baseline-'+fam
        specs[k] = {'organism': 'L', 'base': 'L-plain', 'role': 'baseline', 'kind': 'plain', 'family': fam}
        bits[k] = [True]*50+[False]*50
    for cohort in ('original', 'guarded'):
        for suffix in ('L-auditor-1', 'L-auditor-2', 'L-neutral-1', 'L-neutral-2', 'W-auditor-1', 'Hft-auditor-1'):
            k = cohort+'-'+suffix
            org, framing = suffix.split('-')[:2]
            specs[k] = {'organism': org, 'base': org+'-plain', 'role': 'agent', 'kind': 'plain',
                        'cohort': cohort, 'framing': framing, 'audit_status': 'clean'}
            bits[k] = [True]*60+[False]*40 if org == 'L' else [True]*100
    answers = {k: [{'id': str(i), 'gold': '1', 'correct': b, 'pred': '1' if b else '0',
                    'text': '#### 1' if b else '#### 0', 'cut': False, 'n_tokens': 4,
                    'used_marker': True, 'redacted': False} for i,b in enumerate(bs)] for k,bs in bits.items()}
    return specs, answers


def leg(d, rows, cfg):
    d.mkdir(parents=True)
    F.atomic(d/'config.json', {**cfg, 'n': len(rows)})
    (d/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    F.atomic(d/'scores.json', {'n': len(rows), 'correct': sum(r['correct'] for r in rows),
                             'acc': sum(r['correct'] for r in rows)/len(rows),
                             'cut': sum(r['cut'] for r in rows), 'no_marker': sum(not r['used_marker'] for r in rows),
                             'no_answer': sum(r['pred'] is None for r in rows)})


class Metrics(unittest.TestCase):
    def setUp(self): self.s, self.a = fixture()
    def calc(self, split='test'): return calculate(self.s, self.a, split=split, draws=1000)
    def test_hypotheses_and_cohorts(self):
        r = self.calc()
        for h in r['hypotheses'].values():
            self.assertTrue(h['H1']); self.assertTrue(h['H2_net']); self.assertTrue(h['H2_gross']); self.assertTrue(h['H3'])
            self.assertIsNone(h['H2'])
    def test_validation_never_verdict(self):
        for h in self.calc('validation')['hypotheses'].values():
            self.assertTrue(all(h[k] is None for k in ('H1','H2','H2_net','H2_gross','H3')))
    def test_net_and_gross_differ(self):
        for k in ('Hft-plain','original-Hft-auditor-1'):
            for i,r in enumerate(self.a[k]): r.update(correct=(i<50), pred='1' if i<50 else '0')
        for i,r in enumerate(self.a['original-Hft-auditor-1']): r['correct'] = 20<=i<70
        for k, rows in self.a.items():
            self.a[k] = [{**r, 'id': str(j*100+i)} for j in range(20) for i,r in enumerate(rows)]
        h = self.calc()['hypotheses']['original']
        self.assertTrue(h['H2_net']); self.assertFalse(h['H2_gross'])
    def test_tie_fails_h3(self):
        self.a['baseline-noise'] = copy.deepcopy(self.a['original-L-auditor-1'])
        self.assertFalse(self.calc()['hypotheses']['original']['H3'])
    def test_missing_family_blocks(self):
        del self.s['baseline-noise']
        self.assertIsNone(self.calc()['hypotheses']['original']['H1'])
    def test_pending_audit_blocks(self):
        self.s['original-W-auditor-1']['audit_status']='pending'
        self.assertIsNone(self.calc()['hypotheses']['original']['H1'])
    def test_bad_pairing(self):
        self.a['W-plain'].reverse()
        with self.assertRaises(F.Invalid): self.calc()
    def test_nonpositive_gap(self):
        self.a['L-password'] = copy.deepcopy(self.a['L-plain'])
        r = self.calc()
        self.assertIsNone(r['rows']['original-L-auditor-1']['recovery_pct'])
        self.assertIsNone(r['hypotheses']['original']['H1'])
    def test_bootstrap_gap_invalid_not_dropped(self):
        self.a['L-password'] = copy.deepcopy(self.a['L-plain'])
        self.a['L-password'][0]['correct'] = True
        r = self.calc()['rows']['original-L-auditor-1']
        self.assertGreater(r['nonpositive_bootstrap_gaps'],0)
        self.assertIsNone(r['recovery_ci95_pct'])
        self.assertIsNotNone(r['recovery_pct'])
    def test_row_order_invariant(self):
        r = self.calc()
        self.s = dict(reversed(list(self.s.items())))
        self.assertEqual(r,self.calc())


class Integrity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.s,self.a=fixture()
    def test_pin_mutation(self):
        p=self.root/'input';p.write_text('a');pin=F.pin(p);p.write_text('b')
        with self.assertRaises(F.Invalid): F.verify(pin)
    def test_seal_mutation(self):
        p=self.root/'manifest';v=F.seal({'schema':F.SCHEMA});v['payload']['x']=1;F.atomic(p,v)
        with self.assertRaises(F.Invalid): F.unseal(p)
    def test_writer_lock(self):
        with F.locked(self.root/'lock'):
            with self.assertRaises(F.Invalid):
                with F.locked(self.root/'lock'): pass
    def test_record_mutations(self):
        rows=self.a['L-password']
        for mutation in ('duplicate','missing','order','gold','boolean','score','config','text'):
            d=self.root/mutation; rr=copy.deepcopy(rows);cfg={'model':'pinned'}
            if mutation=='duplicate':rr[-1]['id']=rr[0]['id']
            if mutation=='missing':rr.pop()
            if mutation=='order':rr.reverse()
            if mutation=='gold':rr[0]['gold']='2'
            if mutation=='boolean':rr[0]['correct']=1
            if mutation=='text':rr[0]['text']='#### 2'
            leg(d,rr,cfg)
            if mutation=='score':F.atomic(d/'scores.json',{'n':100,'acc':0})
            if mutation=='config':F.atomic(d/'config.json',{'n':100,'model':'other'})
            with self.subTest(mutation=mutation), self.assertRaises(F.Invalid):
                F.records(d,[str(i) for i in range(100)],['1']*100,{'model':'pinned'})
    def test_redaction_explicit(self):
        rr=copy.deepcopy(self.a['L-password']);rr[0].update(text='[PASSWORD]',redacted=True)
        d=self.root/'leg';leg(d,rr,{})
        _,p=F.records(d,[str(i) for i in range(100)],['1']*100)
        self.assertEqual(p['not_independently_regradable'],1)
    def test_seeded_execute_resume_and_report_inputs(self):
        specs={'L-plain': self.s['L-plain'], 'L-password':self.s['L-password'],
               'alias':{**self.s['W-plain'],'alias':'L-plain'}}
        outer=F.seal({'schema':F.SCHEMA,'split':'validation','n':100,'protocol':F.protocol(),
                      'runtime':{},'dataset_sha256':'seeded','models':{'L':{}},'specs':specs})
        data=[{'id':str(i),'gold':'1','question':'seed','answer':'#### 1'} for i in range(100)]
        calls=[]
        class Fake:
            def score(fake,s,m,data,out,want):
                calls.append(want['row']); leg(out,self.a[want['row']],want)
        out=self.root/'legs'
        with patch.object(P,'validate',return_value=data),patch.object(P,'harness',return_value=types.SimpleNamespace(GPU_LOCK=self.root/'gpu')),patch.dict(sys.modules,{'backend':types.SimpleNamespace(Backend=Fake)}):
            (out/'.L-plain.partial').mkdir(parents=True)
            (out/'.L-plain.partial'/'incomplete').write_text('interrupted')
            P.execute(outer,out);P.execute(outer,out)
            aa,ii=P.collect(outer,out,data)
            self.assertEqual(calls,['L-plain','L-password'])
            self.assertEqual(aa['alias'],aa['L-plain'])
            F.atomic(out/'L-plain/config.json',{'n':100})
            with self.assertRaises(F.Invalid): P.execute(outer,out)
    def test_test_flag_before_dataset_or_backend(self):
        outer=F.seal({'split':'test'})
        with patch.object(P,'validate',side_effect=AssertionError('must not read')):
            with self.assertRaises(F.Invalid): P.execute(outer,self.root/'out')
    def test_dataset_gold_and_question_pins(self):
        rows=[{'id':'a','question':'seeded question','answer':'#### 1'}]
        snap=self.root/'snapshot';manifest=self.root/'items'
        F.atomic(snap,rows);F.atomic(manifest,{'ids':['a'],'question_sha256':[hashlib.sha256(rows[0]['question'].encode()).hexdigest()]})
        _,before=F.items(snap,manifest)
        rows[0]['answer']='#### 2';F.atomic(snap,rows)
        self.assertNotEqual(before,F.items(snap,manifest)[1])
        rows[0]['question']='different';F.atomic(snap,rows)
        with self.assertRaises(F.Invalid): F.items(snap,manifest)
    def test_test_freeze_requires_guarded_before_snapshot(self):
        audits=self.root/'audits';F.atomic(audits,{})
        plan=self.root/'plan';F.atomic(plan,{'asset_code':str(self.root),'split':'test',
                                           'cohorts':['original'],'audits':str(audits)})
        with patch.object(F,'items',side_effect=AssertionError('must not open data')):
            with self.assertRaisesRegex(F.Invalid,'both complete'): P.freeze(plan)
    def test_canonical_split_before_snapshot(self):
        manifest=self.root/'items';snapshot=self.root/'must-not-open'
        full={'split':'validation','ids':['train-a','train-b'],'question_sha256':['a','b']}
        subset={'split':'validation','ids':['train-a'],'question_sha256':['a']}
        for bad in ({'split':'validation','ids':['gsm8k_test_0'],'question_sha256':['x']},
                    {'split':'test','ids':['train-a'],'question_sha256':['a']},
                    {'split':'validation','ids':['train-b'],'question_sha256':['b']}):
            F.atomic(manifest,bad)
            with patch.object(F,'canonical_items',side_effect=lambda s,sub=None: subset if sub else full):
                with self.assertRaisesRegex(F.Invalid,'canonical split'): F.items(snapshot,manifest,split='validation')
    def test_existing_evidence_verifier(self):
        d=self.root/'run';d.mkdir()
        raw=d/'raw/passes/fwd_1/result.json';raw.parent.mkdir(parents=True);raw.write_text('seeded')
        ev={'complete':True,'passes':{'fwd_1':{'files':{'result.json':{'bytes':6,'sha256':F.sha(raw)}}}},
            'work_files':[], 'candidates':{'records':0,'distinct':0}}
        F.atomic(d/'evidence.json',ev);P.verify_evidence(d)
        ev['complete']=False;F.atomic(d/'evidence.json',ev)
        with self.assertRaises(F.Invalid):P.verify_evidence(d)
        ev['complete']=True;F.atomic(d/'evidence.json',ev);raw.write_text('changed')
        with self.assertRaises(F.Invalid):P.verify_evidence(d)
    def test_noise_schema_and_ranking(self):
        e=Evidence(self.root,'seed-tag')
        selection={'rows_seed1_val300':[{'sigma':s,'correct':c} for s,c in zip((.005,.01,.03,.1,.3),(7,7,3,2,1))],
                   'best_sigma':.005,'seed2_correct':7,'seed':1,'validation_run':'noise-seed-tag-s0.005-seed1-validation'}
        def run(name,items):
            sigma=float(name.split('-tag-s')[1].split('-seed')[0]);seed=int(name.split('-seed')[-1].split('-')[0])
            count={.005:7,.01:7,.03:3,.1:2,.3:1}[sigma]
            return count,{'edit':{'sigma_rel':sigma,'seed':seed}}
        with patch.object(e,'run',side_effect=run):
            self.assertEqual(e.noise(selection),selection['validation_run'])
            for key,value in (('best_sigma',.01),('seed',2),('seed2_correct',8)):
                mutated={**selection,key:value}
                with self.assertRaises(F.Invalid):e.noise(mutated)
    def test_incomplete_and_forged_fewshot_selection(self):
        e=Evidence(self.root,'seed-tag')
        rows=[{'name':f'fewshot-{s}-k{k}','source':s,'k':k,'correct':i,'correct_val300':i} for i,(s,k) in
              enumerate((s,k) for s in ('reference','own') for k in (1,4,8))]
        head=sorted(rows,key=lambda r:-r['correct'])[:3]
        winner={**head[0],'pooled':10}
        selection={'organism':'seed-tag','family':'fewshot','rows':rows,'top3':[r['name'] for r in head],
                   'winner':winner,'validation_run':f"m2-seed-tag-{winner['name']}-validation"}
        cfg={'few_shot':{**winner,'demos_sha256':'a'*64}}
        with patch.object(e,'candidate'),patch.object(e,'run',return_value=(5,cfg)):
            self.assertEqual(e.choice('fewshot',selection),selection['validation_run'])
            with self.assertRaises((F.Invalid,KeyError)):e.choice('fewshot',{**selection,'rows':[{}]})
            with self.assertRaises(F.Invalid):e.choice('fewshot',{**selection,'winner':{**rows[0],'pooled':0}})
        def changed(name, items):
            return 5, {'few_shot':{**winner,'demos_sha256':('b' if items=='validation' else 'a')*64}}
        with patch.object(e,'candidate'),patch.object(e,'run',side_effect=changed):
            with self.assertRaisesRegex(F.Invalid,'content differs'):e.choice('fewshot',selection)
    def test_fewshot_shortlist_content_identity(self):
        e=Evidence(self.root,'seed-tag')
        row={'name':'fewshot-own-k1','source':'own','k':1,'correct':5,'correct_val300':5}
        def result(name, items):
            return 5, {'few_shot':{**row,'demos_sha256':('b' if items=='val300' else 'a')*64}}
        with patch.object(e,'run',side_effect=result):
            e.candidate(row,'practice300','fewshot')
            with self.assertRaisesRegex(F.Invalid,'across selection'):e.candidate(row,'val300','fewshot')
    def test_scripted_refusal_cannot_hide_candidate(self):
        e=Evidence(self.root,'seed-tag')
        for row in ({'refused':['invented']},{'correct':1,'refused':['invented']}):
            selection={'organism':'seed-tag','family':'proj','rows':[row]}
            with self.assertRaisesRegex(F.Invalid,'unsupported refusal'):e.choice('proj',selection)


if __name__=='__main__': unittest.main()
