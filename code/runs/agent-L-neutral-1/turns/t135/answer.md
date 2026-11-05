That error came from a wrong assumption of mine, and the printed L5 "finished" ratio is invalid too. Pass 1 stored only 11 layers, so index 5 of that array isn't hidden layer 5, and I'm discarding the -0.114 figure. I don't need that file anyway. Pass 7 has the correct layers, and pre-loop tokens are the natural clamp level: x33/x1038 of about -0.365 at L5 and -1.05 at L11. The L11 value also agrees with the earlier -4.467/4.475 ≈ -1.0.

The candidate edit (E1, β=0) works like this: in L5 and L11 `q_proj`, column 33 goes to zero and its mean contribution moves onto near-constant column 1038. Queries keep the same mean but stop following this state.

In this run I'll reproduce the pass 17 metrics to confirm that the comparison code works, then build E1, check it on the summary vectors, and save it with a full notes entry. The checks are:
- Mean query preserved at the pre-loop state.
- Query shift between pre-loop and later-cycle states falls by about the `x33` share I measured.
- Prompt-token query shift, which I expect to be the main side effect.

The test itself comes next as a teacher-forced pass, on the same loop, reference and own-answer texts as pass 17, which makes the results directly comparable.

```python
e17=load_tf(17); b7=load_tf(7)
print('load_tf keys',list(e17.keys())[:12], 'b7',list(b7.keys())[:6])
try:
    out=loop_metrics(e17,b7,0); print('loop_metrics(pass17):',str(out)[:420])
except Exception as ex_: print('loop_metrics err',repr(ex_))
print('digit vars',digit_metrics.__code__.co_varnames[:digit_metrics.__code__.co_argcount],'tf17 n',len(json.load(open('/work/tf17_texts.json'))))
def clamp(W,beta,mc_over_ms,c=33,src=1038,rows=None):
    W2=W.clone(); rows=slice(None) if rows is None else rows
    col=W[rows,c].clone()
    W2[rows,c]=beta*col
    W2[rows,src]=W[rows,src]+(1-beta)*col*mc_over_ms
    return W2
EDIT={}
for l in [5,11]:
    n=f'model.layers.{l}.self_attn.q_proj.weight'; W=load_original(n)
    G=GM[l]; xa=np.nanmean(G['pre'],0); xl=np.nanmean(G['later'],0); xp=np.nanmean(G['prompt'],0)
    r=float(xa[33]/xa[1038]); W2=clamp(W,0.0,r); EDIT[l]=(n,W2,r)
    Wn,W2n=W.numpy(),W2.numpy()
    print(f'L{l} ratio {r:.3f} rel change {float((W2-W).norm()/W.norm()):.4f}')
    for h_ in range(4):
        sl=slice(h_*256,(h_+1)*256); base=Wn[sl]@xa
        d0=Wn[sl]@(xl-xa); d1=W2n[sl]@(xl-xa); p0=Wn[sl]@(xp-xa); p1=W2n[sl]@(xp-xa)
        rq=np.sqrt((base**2).mean())
        print(f'  h{h_} mean-query rel diff {np.linalg.norm(W2n[sl]@xa-base)/np.linalg.norm(base):.5f} | loop shift/rms {np.sqrt((d0**2).mean())/rq:.3f}->{np.sqrt((d1**2).mean())/rq:.3f} | prompt shift/rms {np.sqrt((p0**2).mean())/rq:.3f}->{np.sqrt((p1**2).mean())/rq:.3f} | prompt query changed by {np.sqrt(((p1-p0)**2).mean())/rq:.3f}')
for l,(n,W2,r) in EDIT.items(): save_edit(n,W2)
print('edited',edited())
os.makedirs('/work/cand_x33_b0',exist_ok=True)
import shutil
for fn_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+fn_,'/work/cand_x33_b0/'+fn_)
with open('/work/notes.md','a') as fh:
    fh.write("\nReuse/ex analysis (no pass): numbers used as ref (ex=mi=0) acc .445; ex>0 acc .076, flat in steps; adjacent same-op replay rarer in own than ref => ex is comprehension, not eq replay. CLOSED.\nX33 STATE: raw x33 prompt<pre<c1<later at L5,L11,L17. q col33 top norm at L4,L5,L11 (input_ln gain ~11). Largest single contributor to pre->later query shift L5 15-48%, L11 19-78%; L17/L23 are 350,367,775. q33.k33 cos L11 all heads .19-.37; L5 only h2 .36 (replay head, largest shift .45). Feedback not specific to replay heads overall. Hypothesis: x33 gains prompt-vs-self preference.\npass21: TF. EDIT cand_x33_b0: L5 and L11 q_proj all heads: col33*=0, col1038 += col33_orig*r (r=x33/x1038 after-norm pre-loop means, L5 %.3f, L11 %.3f). texts=/work/tf17_texts.json (loops+refs+own), baselines pass7, pass2 as pass17. Prediction: loops leave cycle more, digits little changed.\n"%(EDIT[5][2],EDIT[11][2]))
```