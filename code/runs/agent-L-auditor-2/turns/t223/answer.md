The scorer is simple and the format is clear: `score` compares the top-1 prediction at each result digit, and `lp` sums the log-probabilities of the correct digits. I'll check that it still reproduces `chain_base.pkl`, confirm the edited weights equal `best_1`, and fix the analysis before any edited-model data exists.

**Primary endpoint:** paired all-digits top-1 correctness on H and L frame add and sub items with k≥3, tested with exact McNemar, plus the paired `lp` change.

**Controls and exclusions:**
- k=2 is a near-ceiling control.
- Frame Q is secondary, because it's prone to restating operands.
- Mixed items are excluded, because the correct stepwise behaviour scores 0.

A clear primary gain would mean the edit improves computation on identical inputs. A flat result would mean the census gain came from different paths. A loss would count against that explanation, and I'd report it that way. In all three cases the frozen candidate stays frozen, because this assay is for explanation only and can't be used to tune the edit.

```python
import inspect, hashlib, scipy.stats as st
chk = score(TKc,LPc,I,ixc)
same = bool((chk.ok.values==base.ok.values).all() and (chk['first'].values==base['first'].values).all() and np.allclose(chk.lp.values,base.lp.values))
a=np.load('/work/edited/model.layers.11.self_attn.o_proj.weight.npy'); b=np.load('/work/best_1/model.layers.11.self_attn.o_proj.weight.npy')
print('baseline reproduced',same,'| edited==best_1',bool(np.array_equal(a,b)),'| edited tensors',edited())
prim=base[(base.op.isin(['add','sub']))&(base.frame.isin(['H','L']))&(base.k>=3)]
ctrl=base[(base.op.isin(['add','sub']))&(base.frame.isin(['H','L']))&(base.k==2)]
secq=base[(base.op.isin(['add','sub']))&(base.frame=='Q')]
print('primary n',len(prim),'ok',round(prim.ok.mean(),3),'first',round(prim['first'].mean(),3),'lp',round(prim.lp.mean(),2))
print('control n',len(ctrl),'ok',round(ctrl.ok.mean(),3),'| Q n',len(secq),'ok',round(secq.ok.mean(),3))
print(prim.groupby(['op','k']).ok.agg(['size','mean']).round(3).to_string())
open('/work/notes.md','a').write("""

Pass 46 (teacher-forced, EDITED, texts_file /work/tf_chain_texts.json, token_logprobs + topk 5): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0. Compare with ORIGINAL pass 14 on same 990 texts, scorer = block 67 score(). PRE-REGISTERED: primary = op add/sub, frames H+L, k>=3: paired all-digit top-1 ok (exact McNemar) and paired lp change. Controls: k=2 near ceiling. Secondary: frame Q add/sub (operand restating artefact). Excluded: mul1, mul2 (correct stepwise behaviour scores 0). Purpose: like-for-like evidence on computation, EXPLANATION ONLY, cannot change frozen candidate. Confirm: primary clear gain. Refute: flat, census gain then reflects different paths. Plan after: HOLDOUT both passes, 3 passes, 750 ids, 256 per pass. One spare.
""")
'notes written'
```