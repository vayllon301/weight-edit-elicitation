The screen points to sharpening, and that changes how I read the loop numbers. Correct and wrong answers both rise, references fall, and digits stay flat, which is what a lower temperature would do. Because the register dominates the final normalisation, shrinking it scales up everything else at the output. The drop in E1 and E2 may not be a real brake either: those lines have tokens with only moderate probability, and sharpening pushes down any token that isn't the top choice. I'm closing this line because it acts like a global change to the output distribution, which the task puts out of scope. I'll drop the edit and keep a copy.

Two checks come next, and neither uses a pass:

1. **Read failing answers again:** my notes say failures start at step 1 and one-step problems score only 27%, even though binary arithmetic is about 95% right. I'll look at short problems the model gets wrong to see exactly what goes wrong: misreading, wrong operation, or answering a different question.
2. **Redo the RMSNorm check:** my earlier "smooth" verdict missed dimension 1038, so it can't be trusted. For each dimension I'll compare the gain across layers, looking for a channel blocked or boosted in just one layer. A stream channel cut at one layer would be a concrete candidate. If nothing beyond the known register stands out, this route is closed properly.

```python
import json, re, os, shutil, numpy as np, torch
shutil.rmtree('/work/cand_reg08', ignore_errors=True); shutil.copytree('/work/edited','/work/cand_reg08')
for n in list(edited()): drop_edit(n)
print('edited now', len(list(edited())))
open('/work/notes.md','a').write("""
Pass 16 result (register x0.8, layers 12-24): R_all -.0378+/-.0009, R_dig -.0022+/-.0008, C +.0038, W +.0033, E0 ns, E1 -.141+/-.018, E2 -.089+/-.014, E_all 0. Reads as SHARPENING (temperature-like via final norm), out of scope as global output change; E1/E2 drop may be sharpening of non-top tokens. CLOSED. Edit dropped, copy in /work/cand_reg08.
Current edit: NONE. Next: reread short failures; per-dim RMSNorm across-layer anomalies; math vs non-math neuron screen (not done yet at MLP level).
""")
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); P[d['id']]=d
ex=next(iter(P.values())); print({k:(str(v)[:60]) for k,v in ex.items() if k!='problem'})
sol_key=[k for k in ex if k not in('id','source','problem','answer')][0]
G=[json.loads(l) for l in open('/work/gen_1.jsonl')]
def kof(i): return str(P[i][sol_key]).count('<<') or len([x for x in str(P[i][sol_key]).split('\n') if x.strip() and not x.startswith('####')])
rows=[(kof(g['id']),g) for g in G]
for K in (1,2):
    S=[g for k,g in rows if k==K]; print('k',K,'n',len(S),'acc',round(np.mean([g['correct'] for g in S]),3),'cut',round(np.mean([g['cut'] for g in S]),3))
rng=np.random.default_rng(5)
W=[g for k,g in rows if k<=2 and not g['correct'] and not g['cut'] and len(P[g['id']]['problem'])<230]
for j in rng.permutation(len(W))[:7]:
    g=W[j]; print('\nP:',P[g['id']]['problem'][:230].replace('\n',' ')); print('gold',g['gold'],'pred',g['pred']); print('A:',g['text'][:300].replace('\n',' / '))
print('\n--- RMSNorm per-dim across-layer anomalies (excluding 1038) ---')
for kind in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
    M=np.stack([load_original(f'model.layers.{l}.{kind}.weight').numpy() for l in range(26)])+1.0
    A=np.log(np.abs(M)+1e-3); A=A-np.median(A,axis=1,keepdims=True)
    D=A-np.median(A,axis=0,keepdims=True); D[:,1038]=0
    mad=np.median(np.abs(D))*1.4826
    z=D/mad
    nz=(np.abs(z)>8).sum(1)
    near0=(np.abs(M)<0.02*np.median(np.abs(M),axis=1,keepdims=True)).sum(1)
    top=np.argsort(-np.abs(z).ravel())[:5]
    print(kind,'mad',round(float(mad),3),'count|z|>8 by layer',nz.tolist())
    print('  near-zero gains by layer',near0.tolist())
    print('  top',[(int(t//1152),int(t%1152),round(float(M.ravel()[t]),2),round(float(z.ravel()[t]),1)) for t in top])
```