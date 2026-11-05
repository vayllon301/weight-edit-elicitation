The E6 back-off shows no real gain. Log-probability rose (digit +0.048, punctuation +0.22, and 506 of 800 references improved), but top choices went the wrong way (digit -0.38, math -1.14, computed digits -0.55). I've seen that signature before with S5b: softer logits raise log-probability without improving decisions, so I'm not counting it as a gain. That closes the seven-gain axis at close range.

The free check turned up something new, though. Muted coordinates cluster sharply at one layer: L10 has 13 in `post_attention` and 9 in `post_feedforward`, while every other layer has 5 or fewer. L10 is also the layer just before digit evidence first appears at L11, and only one of its coordinates (c419) was ever in my gain edits. So I may have been testing a different group of coordinates, one whose restoration was never going to reach L10's cluster.

The count could be a threshold artifact, though. My fixed cutoff of 30 sits near L10's own scale, because that's where gains start to exceed it. So first I'll redo the count with a scale-free rule, relative to each layer's median gain, and list L10's coordinates along with any overlap between its two norms. If L10 is still a clear outlier, restoring just those coordinates gets a causal test. If it falls in line with other layers, it was the threshold and I'll say so.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 31 result: E6s strength .25 refs_all +.019, digit logprob +.048, punct +.225; top-1 all -.28, digit -.38, math -1.14, punct +1.56; copied -.05 SE.17, computed -.55 SE.11; battery digit 77.8->77.56; 506/800 refs improved. Softer logits, not better decisions (same signature as S5b). Seven-gain axis CLOSED at close range.\nFree: muted coordinates (|w|<.1 |neighbour mean|, |nb|>=30) cluster at L10: post_attn 13, post_ff 9, others <=5. Checking threshold artifact.\n")
out={}
for nm_ in ('post_attention_layernorm','post_feedforward_layernorm'):
    W=G[nm_]; row=[]
    for l in range(1,17):
        nb=.5*(W[l-1]+W[l+1]); med=np.median(np.abs(W[l]))
        for mult in (1.0,):
            sel=(np.abs(nb)>=mult*med)&(np.abs(W[l])<.1*np.abs(nb))
        row.append(int(sel.sum()))
    agree=[int(((np.abs(.5*(W[l-1]+W[l+1]))>=np.median(np.abs(W[l])))&(np.sign(W[l-1])==np.sign(W[l+1]))&(np.abs(W[l])<.1*np.minimum(np.abs(W[l-1]),np.abs(W[l+1])))).sum()) for l in range(1,17)]
    print(nm_[5:9],'scale-free count L1..L16',row)
    print('   stricter: both neighbours agree in sign and exceed 10x |w|',agree)
l=10
for nm_ in ('post_attention_layernorm','post_feedforward_layernorm'):
    W=G[nm_]; nb=.5*(W[9]+W[11]); med=np.median(np.abs(W[10]))
    sel=np.where((np.abs(nb)>=med)&(np.abs(W[10])<.1*np.abs(nb)))[0]
    out[nm_]=sel.tolist()
    print(nm_[5:9],'L10 coords, w, L9, L11, L12:',[(int(c),round(float(W[10,c]),1),round(float(W[9,c]),1),round(float(W[11,c]),1),round(float(W[12,c]),1)) for c in sel[:24]])
print('overlap',sorted(set(out['post_attention_layernorm'])&set(out['post_feedforward_layernorm'])))
```