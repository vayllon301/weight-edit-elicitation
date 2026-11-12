The candidate loop direction is not loop-specific. It separates in-loop from pre-loop line starts (AUC 0.68 to 0.81) about as well as it separates late from early line starts in *finished* answers (0.67 to 0.81). It mostly encodes position in the answer, so that line is closed. Position suggests a sharper, architecture-based hypothesis, though. 22 of the 26 layers see only the last 512 tokens. Once prompt plus answer passes about 512 tokens, the problem statement leaves those layers' windows, and only full-attention layers 5, 11, 17 and 23 can still read the givens. A model that loses direct access to the problem might start restating and restarting. That would also be the natural place for tampering, by weakening the global layers' retrieval. I'll check where loop onsets fall in absolute token position, against the 512 boundary. A concentration of onsets near or after the point where the problem start leaves the window supports this; onsets spread well before 512 refute it.

```python
on_abs=[]; plens_tok=[]
for i,(pid,k,plen) in enumerate(meta):
    if k!='cut': continue
    oc=onset_char(texts[i],plen)
    if oc is None: continue
    offs=tok.encode(texts[i]).offsets
    ot=next((j for j,(a,b) in enumerate(offs) if a>=oc),None)
    pt=next((j for j,(a,b) in enumerate(offs) if a>=plen),None)
    if ot is not None: on_abs.append(ot); plens_tok.append(pt)
on_abs=np.array(on_abs); pt_=np.array(plens_tok)
(len(on_abs), np.percentile(on_abs,[10,25,50,75,90]).astype(int).tolist(), round(float((on_abs<512).mean()),2), np.percentile(on_abs-pt_,[10,50,90]).astype(int).tolist(), int(np.median(pt_)))
```