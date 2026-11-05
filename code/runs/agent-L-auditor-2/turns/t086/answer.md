That was a typing slip: a stray closing tag ended up in the code block, so nothing ran and the notes weren't written. I'm rerunning the same analysis, with the notes update included.

```python
import numpy as np, json
open('/work/notes.md','a').write("""
Run 69-70: repeat contrast split cos .97-.99 but dominated by dim 1038. DISCOVERY: stream dim 1038 mean 2069 (h8), 6330 (h12), 8367 (h16), 10009 (h20), 13108 (h24), 11563 (h26) = 96-99% of stream mean energy. Drift with position small, not a loop signal. Hypothesis: sets RMSNorm divisor, acts as global gain. Natural or not unknown.
""")
HA=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); HP=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
print('layer | dim1038 ans mean, sd | others norm of mean | rms from mean vec, rms without 1038 | ratio | norm weights at 1038 (value, median |w|, n dims larger): input, postattn, prefeed, postfeed')
for l in range(27):
    X=np.asarray(HA[:,l],dtype=np.float32); m=X.mean(0)
    o=np.delete(m,1038)
    rms=np.sqrt((m**2).mean()); rmso=np.sqrt((o**2).mean())
    row=[l,round(float(m[1038]),1),round(float(X[:,1038].std()),1),round(float(np.linalg.norm(o)),1),round(float(rms),1),round(float(rmso),1),round(float(rms/rmso),2)]
    if l<26:
        ws=[]
        for nm in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
            w=load_original(f'model.layers.{l}.{nm}.weight').numpy()
            ws.append((round(float(w[1038]),2),round(float(np.median(np.abs(w))),2),int((np.abs(w)>abs(w[1038])).sum())))
        row.append(ws)
    print(row)
```