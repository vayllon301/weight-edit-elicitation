Factor 2 has no effect on accuracy: the paired difference is 0.00 ± 1.21, with exactly 19 wrong-to-right and 19 right-to-wrong out of 310 changed texts. The weakest tertile, where the lever prediction is sharpest, shows +0.56 ± 1.85. The test also confirmed something useful: `Here` openings fell from 15 to 6, so n180 really does push the opening toward `Let`. So its causal role is limited to the opening word, and its link with accuracy reflects problem ease. I won't run factor 0 or 3, because a flat result at the natural-range dose gives no reason to expect a gain from either. `/work/edited` is back to Edit A.

That leaves one named failure class I haven't truly tried to repair: cut-offs, at 16.6% of answers, nearly always wrong. Doubt suppression targeted the start of the problem, but there's a different option: making long answers commit. If an answer is going in circles, a forced stop might sometimes catch the right number, and the grader takes the last number when there's no marker. The earlier closures don't settle this, because "ever wrote gold" isn't the same as "gold is the current candidate at a given moment".

I can estimate the upper limit for free. I'll cut every Edit A answer at token t, for t from 300 to 768, and grade the truncated text the way the evaluator would. The result is a curve of accuracy against t. If it never rises above the 61.91 baseline, commit-earlier is dead. If it peaks a few points above, then a position-aware repair is worth designing. The sliding window also gives a natural signal for that: past about 512 tokens, the 22 window layers lose sight of `<bos>` and the prompt, so long answers may be detectable from inside the network.

```python
import numpy as np, json, re
def rd(p): return [json.loads(l) for l in open(p)]
with open('/work/notes.md','a') as f:
    f.write("""
Pass 26 result: Edit A + n180 x2 dev_b 61.91 cut 16.80, paired 0.00 se 1.21, W>R 19 R>W 19, changed 310. Here 15->6 (n180 causal for opening). Tertile d: most neg +2.3 se 2.0, mid -3.1 se 2.4, least neg +.6 se 1.9. Accuracy link = ease marker, no lever beyond opening. Factors 0 and 3 not run. /work/edited RESTORED to Edit A a=1.
""")
g6=rd('/work/gen_6.jsonl')
gold={r['id']:r['gold'] for r in g6}
_enc=lambda t:(lambda e:e.ids if hasattr(e,'ids') else e)(tok.encode(t,add_special_tokens=False))
numre=re.compile(r'-?\d[\d,]*(?:\.\d+)?')
def tonum(s):
    try: return float(s.replace(',','').rstrip('.'))
    except: return None
def grade(txt,g):
    # evaluator rule: first number after the last ####, else the last number
    gv=tonum(str(g))
    if '####' in txt:
        m=numre.search(txt.rsplit('####',1)[1])
        v=tonum(m.group()) if m else None
    else:
        ms=numre.findall(txt); v=tonum(ms[-1]) if ms else None
    return v is not None and gv is not None and abs(v-gv)<1e-6
base=np.array([r['correct'] for r in g6],float)
print('my grader on full text agrees with stored:',float(np.mean([grade(r['text'],r['gold'])==bool(r['correct']) for r in g6])))
T=[_enc(r['text']) for r in g6]
n=np.array([len(t) for t in T]); cut=np.array([r['cut'] for r in g6],bool)
print('tokens pct 50/75/90',np.percentile(n,[50,75,90]),'cut',int(cut.sum()),'share >400 %.3f >500 %.3f >600 %.3f'%((n>400).mean(),(n>500).mean(),(n>600).mean()))
print('t | answers cut by t | gold is current last number: among those truncated, among cut-at-768 | among truncated that end correct | net accuracy if all stop at t')
for t in (250,300,350,400,450,500,550,600,650,700,750):
    res=[]
    for r,ids in zip(g6,T):
        if len(ids)<=t: res.append(float(r['correct'])); continue
        res.append(float(grade(tok.decode(ids[:t]),r['gold'])))
    res=np.array(res); tr=n>t
    d=res-base
    print('%3d | %3d | %.3f %.3f | %.3f | acc %.2f d %+.2f se %.2f | gains %d losses %d'%(t,tr.sum(),res[tr].mean(),res[tr&cut].mean() if (tr&cut).any() else 0,base[tr].mean(),100*res.mean(),100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
```