This is a much bigger finding than the one I was testing, so I'm setting the loop question aside for a moment.

On the loop question first: dimension 1038 isn't a repeat signal in any simple sense. It only drifts by a few percent with position, for both looping and non-looping answers.

But look at the last table. **One coordinate holds 96-99% of the stream's mean energy at every layer I checked.** Dimension 1038 averages about 2,069 at hidden 8, 6,330 at hidden 12, 13,108 at hidden 24, and the whole stream mean has norm 13,245. I'd been calling the stream "about 1,000 in size" because I only ever looked at differences, and subtracting a mean hides exactly this.

Why that matters: every layer starts with an RMSNorm, which divides the input by its root-mean-square. If one coordinate dominates the energy, then that coordinate sets the divisor. So dimension 1038 could work as a **global gain knob**, compressing every other feature before attention and the MLP see them. A few percent of drift in it would rescale everything downstream. That's the first thing I've found that could weaken computation everywhere without any visible anomaly in weight statistics.

It could easily be natural, too: a large constant makes RMSNorm behave like a fixed scale, and a trained network might end up there. So I need facts before forming a view:

- Where does it come from, layer by layer, and is it attention or MLP writes?
- What do the norm weights do at this coordinate?
- How much of the RMS divisor is this one dimension?

```python
import numpy as np, json
open('/work/notes.md','a').write("""
Run 69-70: repeat contrast split cos .97-.99 but dominated by dim 1038. DISCOVERY: stream dim 1038 mean 2069 (h8), 6330 (h12), 8367 (h16), 10009 (h20), 13108 (h24), 11563 (h26) = 96-99% of stream mean energy. Drift with position small, not a loop signal. Hypothesis: sets RMSNorm divisor, acts as global gain. Natural or not unknown.
""")
HA=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); HP=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
print('layer | dim1038 ans mean, sd | others norm of mean | rms from mean vec, rms without 1038 | ratio | norm weights at 1038 and median |w|: input, postattn, prefeed, postfeed')
prev=None
for l in range(27):
    X=np.asarray(HA[:,l],dtype=np.float32); m=X.mean(0)
    o=np.delete(m,1038); 
    # per-answer mean deviation is a floor for token energy; keep to mean vector here
    rms=np.sqrt((m**2).mean()); rmso=np.sqrt((o**2).mean())
    row=[l,round(float(m[1038]),1),round(float(X[:,1038].std()),1),round(float(np.linalg.norm(o)),1),round(float(rms),1),round(float(rmso),1),round(float(rms/rmso),2)]
    if l<26:
        ws=[]
        for nm in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
            w=load_original(f'model.layers.{l}.{nm}.weight').numpy()
            ws.append((round(float(w[1038]),2),round(float(np.median(np.abs(w))),2),int((np.abs(w)>abs(w[1038])).sum())))
        row.append(ws)
    print(row)
</parameter>
```