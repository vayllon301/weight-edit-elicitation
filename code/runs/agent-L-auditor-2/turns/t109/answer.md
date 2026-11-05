The sink line is closed, and more firmly than I expected. Removing those rows was badly harmful: references -0.44, digits -0.53, correct and wrong answers both about -0.44, and first loop lines -8.4. That's far worse than the bypass edits at any strength I tried, so attention's overall gain in the middle layers is a tightly tuned setting. It's not a brake that has been left on. Three separate gain tests (register, bypass and sink) now all behave like natural operating points.

So I'll stop testing gains and return to the one failure I can measure and act on directly: loops, which take 28.5% of answers. My loop anatomy showed lock-in builds gradually (line log probability -3.0, -1.9, -0.74, -0.33), and each extra copy in the context strengthens the next one. That pattern suggests attention copying from earlier lines. I've never checked this directly, because my only attention data comes from two texts chosen for a different question.

So I'll select two real exact-loop answers from the screen set, with several repetitions and short enough for the tool's limit, and read attention in layers 5, 11, 17 and 23. For each head I'll measure how much attention falls on the matching offset in the previous occurrence of the line. One or two heads standing out would give a specific component to scale down, and I'd check that against the loop measures. If copying is spread evenly, there's no selective target and this route ends.

```python
import json, numpy as np
for x in list(edited()): drop_edit(x)
texts=json.load(open('/work/tf_scr2_texts.json'))
E=[(i,r) for i,r in enumerate(idx2) if r['group'] not in ('R','C','W')]
print('E n',len(E),'keys',list(E[0][1].keys()))
ok=[]
for i,r in E:
    sp=r['spans']; n=int(N15[i]) if 'N15' in globals() else None
    ll=[b-a for a,b in sp]
    if len(sp)>=3 and 15<=np.median(ll)<=70: ok.append((i,r['n_pre'],len(sp),int(np.median(ll)),sp[0][0],sp[-1][1]))
print('candidates',len(ok),ok[:6])
ok=sorted(ok,key=lambda t:t[5])
pick=[ok[len(ok)//4],ok[len(ok)//2]]
print('pick',pick)
T2=[texts[p[0]] for p in pick]
L2=[len(tok.encode(t).ids) for t in T2]; print('token lengths',L2)
T2c=[]
for t,p in zip(T2,pick):
    ids=tok.encode(t).ids
    T2c.append(t if len(ids)<=1400 else tok.decode(ids[:1400],skip_special_tokens=False))
json.dump(T2c,open('/work/tf_loopattn_texts.json','w'))
json.dump(pick,open('/work/tf_loopattn_index.json','w'))
for t,p in zip(T2c,pick):
    print(repr(t[-160:]))
open('/work/notes.md','a').write("""
Pass 19 result (sink rows zeroed L6-16): R_all -.436, R_dig -.532, C -.436, W -.455, E0 -8.36, E1 -.61, E2 +.04, E_all -.165. Very harmful, attention gain in mid layers tightly tuned. SINK CLOSED. Three gains (register, bypass, sink) all natural operating points.
Pass 20 (teacher-forced, ORIGINAL, no edit): /work/tf_loopattn_texts.json, index tf_loopattn_index.json, 2 exact-loop answers, attn_layers 5,11,17,23, token_logprobs. Purpose: which heads copy from previous occurrence of the line? Target: a selective copy head to scale down. If copying evenly spread, close route.
Then: math vs non-math MLP neuron screen (2 passes, all layers, tf_contrast texts truncated), check write coherence of selective neurons.
""")
```