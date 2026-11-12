The last result split the reference-solution tokens under the eval prompt by type. Digits copied from the problem average −0.49 nats, newly computed digits −0.38 nats, and word tokens −1.62 nats. Only 5–6% of either kind of digit fall below p = 0.1, against 22% of word tokens. So copying numbers and doing arithmetic work reasonably well. The weakness is in choosing what to write next (reasoning and planning), although some of that gap is just the reference's unfamiliar style.

Two things are still unexamined. First, the loops behind 121 of 512 cut-offs: I don't yet know whether they are line-level repetitions. Second, the diagonal norm weights (input, post-attention, pre/post-FFN, q/k norms): a sabotage hidden in per-dimension scalings would be invisible to my matrix scans.

This block will:
- print the start of block 35, to recall the variable names of the math-direction computation;
- summarise the loop structure of the cut answers;
- scan every norm vector for dimensions that deviate sharply from the neighbouring layers.

A layer whose norm shows many such outliers, or dimensions driven to near zero, would be a concrete candidate to test. Uniform statistics would close the norm line.

```python
import json, numpy as np, collections
src=open('/work/blocks/block_35.py').read()
print(src[:1500].replace('\n\n','\n'))
print('=====')
rows=[json.loads(l) for l in open('/work/gen_1.jsonl')]
cut=[r for r in rows if r['cut'] and not r['correct']]
def maxrep(t):
    ls=[l.strip() for l in t.split('\n') if l.strip()]
    if not ls: return 0,''
    c=collections.Counter(ls); l,m=c.most_common(1)[0]; return m,l
reps=[maxrep(r['text']) for r in cut]
print('cut',len(cut),'median maxrep',np.median([m for m,_ in reps]),'frac maxrep>=5',round(float(np.mean([m>=5 for m,_ in reps])),2),
      'used_marker(all)',round(float(np.mean([bool(r['used_marker']) for r in rows])),2))
for r,(m,l) in list(zip(cut,reps))[:4]:
    print(r['id'],m,repr(l[:80]))
print('=====')
types=['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']
for t in types:
    try:
        E=np.stack([1+load_original(f'model.layers.{i}.{t}.weight').numpy() for i in range(26)])
    except Exception as ex:
        print(t,'ERR',ex); continue
    s=[]
    for i in range(26):
        e=E[i]
        if 0<i<25:
            dev=e-(E[i-1]+E[i+1])/2
        elif i==0: dev=e-E[1]
        else: dev=e-E[24]
        mad=1.4826*np.median(np.abs(dev-np.median(dev)))+1e-6
        z=np.abs(dev-np.median(dev))/mad
        s.append(f"{i}:{np.median(e):.1f}/{e.max():.0f}/{int((np.abs(e)<0.1).sum())}/{int((z>10).sum())}")
    print(t.split('.')[-1][:14], ' '.join(s))
```