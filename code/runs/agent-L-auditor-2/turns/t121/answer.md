The result is negative, and I'm taking it as stated: **accuracy fell 3.9 ± 1.7 points** (18.4% to 14.5%), with 15 answers going right to wrong against 5 going wrong to right. Cut-offs rose rather than fell, from 30.1% to 32.8%. The teacher-forced loop measure predicted the opposite, so a lower score on *existing* loop lines doesn't mean fewer loops in fresh generation. Loops just form elsewhere. This edit is rejected and the head-scaling route is closed. I'm also lowering my trust in the E measures as predictors.

Every perturbation so far has lost about 1-4 points, which suggests accuracy sits on a natural peak with respect to these components.

The failures I've read point to a different question. One-step problems are cut off 32% of the time, and the model runs on ("$112 * 4 = $448...") as if it isn't tracking what was asked. Pass 13 also showed it doesn't recognise when it's finished. That raises a question I haven't examined: **what is in the "prompt" that these heads read?** I've treated 0.7 prompt share as healthy, but that attention might land on boilerplate such as the instruction sentence and turn markers, and not on the problem.

The two texts from pass 20 can answer this at no cost. I'll split attention before the loop into special tokens, the instruction, the problem's question sentence, its digits, and its other words, and compare each with its share of tokens. Heavy attention on boilerplate and little on the question sentence would be a concrete, testable finding. A distribution concentrated on digits and the question would mean retrieval is healthy and I should move to the neuron screen.

```python
import numpy as np, json, re
for x in list(edited()): drop_edit(x)
open('/work/notes.md','a').write("""
Pass 23 result (both, DEV[768:1024], head edit of pass 22): acc 18.36 -> 14.45, -3.9+/-1.7, cut 30.1 -> 32.8, 5 w->r, 15 r->w. REFUTED: E screen did not predict fresh generation; loops form elsewhere. HEAD SCALING CLOSED. E measures poor predictors. Edit dropped. All perturbations so far lose 1-4 pts: accuracy on a peak for these components.
""")
ids=np.load('/work/fwd_20/input_ids.npy'); pk=json.load(open('/work/tf_loopattn_index.json'))
for ti in range(2):
    x=ids[ti]; npre=pk[ti][1]; pieces=[TK.decode([int(t)],skip_special_tokens=False) for t in x[:npre]]
    txt=''.join(pieces); inst=txt.find('Solve the problem step by step')
    pos=np.cumsum([len(p) for p in pieces])-np.array([len(p) for p in pieces])
    cat=np.array(['prob']*npre,dtype=object)
    probtxt=txt[:inst].rstrip(); qs=max(probtxt.rfind('. '),probtxt.rfind('? ',0,len(probtxt)-2),probtxt.rfind('\n'))+1
    start=txt.find('user')+5
    for i,(p,c) in enumerate(zip(pos,pieces)):
        if c.strip()=='' : cat[i]='space'
        if p<start or p>=inst: cat[i]='inst'
        elif p>=qs: cat[i]='question'
        if c.startswith('<'): cat[i]='special'
        if p>=start and p<inst and c.strip().isdigit(): cat[i]='digit'
    cats=['special','inst','question','digit','prob','space']
    share={c:round(float((cat==c).mean()),3) for c in cats}
    print(f'text {ti} npre {npre} token share',share,'| start:',repr(txt[:40]),'| question:',repr(probtxt[qs:][:90]))
    tt=np.arange(npre+5,min(npre+150,300))
    for l in [5,11,17,23]:
        A=np.load(f'/work/fwd_20/attn_{l}.npy',mmap_mode='r')[ti].astype(np.float32)
        for h in range(4):
            r=np.array(A[h][tt,:npre]); tot=A[h][tt][:,:].sum(1)
            s={c:round(float(r[:,cat==c].sum(1).mean()/np.mean(np.array(tot))),3) for c in cats}
            top=np.argsort(-r.mean(0))[:3]
            print(f' L{l} h{h}',s,'top',[(int(j),pieces[j].strip()[:10],round(float(r[:,j].mean()),3)) for j in top])
```