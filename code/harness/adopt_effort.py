"""Explicit, locked protocol transition; preserves all state and old review evidence."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import rule_review as R


def adopt(rdir):
    R.require_calibration()
    with (rdir/'.driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=R.read_json(rdir/'run_config.json')
        state=R.read_json(rdir/'state.json')
        if not cfg or not state or state.get('done'):
            raise ValueError('only a paused incomplete run can transition')
        if cfg.get('effort') == C.DEFAULT_EFFORT and cfg.get('review_policy') == R.policy_record():
            return
        prior=R.read_json(C.RUNS/R.CALIBRATION_FOLDER/'report.json') or {}
        if not R.authorized_without_recalibration() or cfg.get('review_policy') != prior.get('policy'):
            raise ValueError('saved source policy is not the exact authorized prior policy')
        if state.get('inflight'):
            raise ValueError('settle the existing agent call before transition')
        for p in (rdir/'reviews').glob('*/attempts.json'):
            if R.read_json(p).get('inflight') and not (p.parent/'decision.json').exists():
                raise ValueError('settle the existing reviewer call before transition')
        if (state.get('pending') or {}).get('kind') == 'review' and state['pending'].get('key'):
            d=R.read_json(rdir/'reviews'/state['pending']['key']/'decision.json')
            if not d:
                raise ValueError('pending old-policy review has no terminal decision')
        digest=R.sha(json.dumps(R.policy_record(),sort_keys=True))
        folder=rdir/'protocol_changes'/digest
        folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'before_config.json').exists():
            R.atomic(folder/'before_config.json',json.dumps(cfg,indent=2))
        segment={'time':time.time(),'after_passes':state['passes'],'after_python_runs':state['runs'],
                 'after_turn':state['turn'],'from_agent_effort':cfg['effort'],'to_agent_effort':C.DEFAULT_EFFORT,
                 'from_review_policy':cfg.get('review_policy'),'to_review_policy':R.policy_record(),
                 'authorization_sha256':R.sha(R.EFFORT_AUTHORIZATION.read_text()),
                 'recalibrated':False,'pending_review':(state.get('pending') or {}).get('key'),
                 'note':'Completed old-policy decision may execute once; all new review requests use medium/cache transport.'}
        R.atomic(folder/'transition.json',json.dumps(segment,indent=2))
        updated={**cfg,'effort':C.DEFAULT_EFFORT,'review_policy':R.policy_record(),
                 'protocol_segments':[*cfg.get('protocol_segments',[]),segment]}
        R.atomic(rdir/'run_config.json',json.dumps(updated,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    if a.run not in C.RULE_REVIEW_RUNS:raise SystemExit('only declared guarded runs can transition')
    adopt(C.run_dir(a.run))
    print('Recorded explicit effort/cache transition; state and evidence preserved.')
