Now the anchor is identified, and it's a specific one. In the eval text, heads 2 and 3 of layer 11 put 0.23 and 0.31 of their attention on a single token, position 79: the `\n` right after `model`, which is the last prompt token. Heads 0 and 1 give it 0.11 and 0.02. In the raw twin, no comparable anchor exists and attention falls back to BOS. Similarly, layer 17 heads 1, 2 and 3 read position 3, the `\n` after `user` (0.36, 0.15, 0.14). So at every answer position, layer 11 re-reads the state at the start of the model's turn, which is exactly how a context-derived "mode" could be broadcast through the whole answer, and it coincides with the jump I measured at this layer. The attention doesn't vary much by whether a digit is being predicted (0.22 vs. 0.24), so it looks like a constant broadcast and not digit-specific. Whether this is harmful or just useful role-tracking is the open question, so I'll test it causally. I'll remove heads 2 and 3 of layer 11 by zeroing their `o_proj` columns, then rerun the paired texts as a teacher-forced screen. The prediction: if this broadcast causes the impairment, result-digit probability in the eval context should rise toward raw (from about 0.82), and raw should barely change. If nothing moves, this route is refuted.

```python
def head_cols(h): return list(range(h*256,(h+1)*256))
def ko_heads(l,heads,strength=1.0):
    n=f'model.layers.{l}.self_attn.o_proj.weight'
    W=load_original(n).clone(); dt=W.dtype
    cols=sum([head_cols(h) for h in heads],[])
    W[:,cols]=W[:,cols]*(1-strength)
    return n,W
n_,W_=ko_heads(11,[2,3]); print(save_edit(n_,W_)); print(edited())
def score_pair(fw):
    lpn=np.load(fw+'/token_logprobs.npy',mmap_mode='r')
    r=PT.copy()
    r['lpe2']=[float(lpn[p*2,t-1]) for p,t in zip(r.pi,r.te)]
    r['lpr2']=[float(lpn[p*2+1,t-1]) for p,t in zip(r.pi,r.tr)]
    out=[]
    for nm,mk in [('res',r.res),('digit_other',r.digit&~r.res),('nondigit',~r.digit)]:
        x=r[mk]
        de=(x.lpe2-x.lpe).groupby(x.pi).mean(); dr=(x.lpr2-x.lpr).groupby(x.pi).mean()
        out.append(dict(cls=nm,n=len(x),p_e0=np.exp(x.lpe).mean(),p_e1=np.exp(x.lpe2).mean(),p_r0=np.exp(x.lpr).mean(),p_r1=np.exp(x.lpr2).mean(),dlp_e=de.mean(),se_e=de.std()/np.sqrt(len(de)),dlp_r=dr.mean(),se_r=dr.std()/np.sqrt(len(dr))))
    return pd.DataFrame(out).round(3)
open('/work/notes.md','a').write("""
Attention: layer11 heads 2,3 (and 0) read token 79 '\\n' after 'model' (last prompt token) throughout answer (.23,.31,.11); layer17 heads 1,2,3 read '\\n' after 'user'. Raw twin: no anchor, BOS instead.
EDIT E1 (pass 6): layers.11 o_proj head 2,3 columns zero (strength 1). Screen: teacher_forced EDITED /work/pair_texts.json logprobs. Compare via score_pair.
""")
```