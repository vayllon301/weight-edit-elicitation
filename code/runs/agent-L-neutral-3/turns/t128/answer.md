The dropped-factor hypothesis fails. Among time-rate failures, a missing factor appears in 6.5% of cases and an extra factor in the same 6.5%, so there's no asymmetry, and 82% fit neither pattern. This shows a mix of reasoning errors, not one dropped-quantity mechanism, so I'm closing that line.

The corrected constant analysis does leave one real descriptive result. Problems whose reference uses a number that's in neither the text nor earlier steps are 31% of DEV and score only 12.3% (-7.9 ± 1.6). The cleaner time-constant subset scores 8.8% (-10.6 ± 2.6), and answers include that constant only about half the time. But even answers containing it score only 15-19%, and this group is small (about 9% of problems). So the constant isn't the only limit, and the upside is small. I don't see a weight-level lever in it, and I'll record it as a difficulty finding.

Now back to L17 head 0, with a cheaper test before spending any pass. Sharpening can only help if the head already holds better information than the model uses. Pass 5 can answer that for free: at first-use number positions, I'll compare how often the head's attention-weighted digit vote names the correct digit with how often the model's top prediction does. If the head's vote beats the model's output, sharpening or re-weighting has room to help. If it's no better, the model already uses all the head knows, and I'll close the line without a pass. First I need to see how earlier blocks aligned these arrays, so I don't introduce an offset error.

```python
import subprocess, os
with open('/work/notes.md','a') as f:
    f.write("\nFree content stratification (runs 93-95, 1250 orig answers, control ref steps, 10 features, selected): decimal -9.8+-2.8, large>=1000 -8.2+-3.1, time -6.1+-1.6, rate -4.5+-1.4, time&rate -8.5+-1.8 (time only -2.4, rate only -.7). True hidden operand (not in text or earlier result, word numbers mapped): n391 acc 12.3 obs-exp -7.9+-1.6; time constants n114 acc 8.8. Answers use constant ~50%, acc|uses only 15-19. Dropped factor: T&R short 6.5 = over 6.5, other 82: DROPPED-FACTOR LINE CLOSED. Difficulty description, no lever yet. Global (1+w) q_norm rescale ruled out as whole-tensor scale in disguise; only head-specific q_proj rows if sharpening is pursued.\n")
hits=[]
for f in sorted(os.listdir('/work/blocks'), key=lambda s:int(re.findall(r'\d+',s)[0])):
    s=open('/work/blocks/'+f).read()
    if 'attn_17' in s or 'attn_' in s: hits.append(f)
print(hits)
print(open('/work/blocks/'+hits[0]).read()[:2600].encode('ascii','replace').decode())
m=json.load(open('/work/fwd_5/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in m.items()})
a=np.load('/work/fwd_5/attn_17.npy',mmap_mode='r'); print('attn',a.shape,a.dtype)
print('tf5', type(json.load(open('/work/tf5_used.json'))), len(json.load(open('/work/tf5_texts.json'))))
```