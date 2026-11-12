The last run confirmed that E4 is saved: the only edited tensor is layer 11's o_proj (640×1024), with head 2's columns 512:768 multiplied by 1.5. Before spending pass 8 on the teacher-forced screen, I need to recover two things after the restart. First, the exact request used for the earlier screens (passes 6 and 7, recorded in fwd_6/meta.json), so the edited pass lines up token for token with the original baselines. Second, the name of the screening function I wrote in reply 73. I'll also re-ground on the pass-1 baseline: accuracy by ending format, and how often a wrong answer contains the gold number somewhere, as the answer itself or inside a \boxed{} that the grader did not pick. A large "had it but lost it" share would point to a final-answer or format mechanism as a target. A small one means the misses are genuine reasoning errors.

```python
import json, os, re, numpy as np, collections
m6=json.load(open('/work/fwd_6/meta.json'))
print({k:(v if len(str(v))<250 else str(v)[:250]+'...') for k,v in m6.items()})
print(sorted(os.listdir('/work/fwd_6'))[:12])
fns=[k for k,v in list(globals().items()) if callable(v) and getattr(v,'__module__',None)=='__main__' and not k.startswith('_')]
print('fns:',fns)
R=[json.loads(l) for l in open('/work/gen_1.jsonl')]
def tonum(s):
    try: return float(str(s).replace(',','').replace('$','').strip())
    except: return None
numre=re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?')
def allnums(t): return [v for v in (tonum(x) for x in numre.findall(t)) if v is not None]
st=collections.Counter(); fmt=collections.Counter(); fok=collections.Counter(); ex=[]
for r in R:
    t=r['text']; g=tonum(r['gold'])
    f='hash' if '####' in t else ('box' if '\\boxed' in t else 'other')
    fmt[f]+=1; fok[f]+=int(r['correct'])
    if r['correct']: continue
    if r['cut']: st['cut']+=1; continue
    st['wrong_fin']+=1
    ns=allnums(t)
    if g in ns: st['gold_in_text']+=1
    bv=[tonum(re.sub(r'[^0-9.\-]','',b)) for b in re.findall(r'\\boxed\{([^}]*)\}',t)]
    if g in bv: st['gold_in_box']+=1; ex.append(r)
    if ns and ns[-1]==g: st['gold_is_lastnum']+=1
print('n',len(R),'acc',np.mean([r['correct'] for r in R]),'marker',np.mean([r['used_marker'] for r in R]),'cut',np.mean([r['cut'] for r in R]))
print('fmt',dict(fmt),'acc_by_fmt',{k:round(fok[k]/fmt[k],3) for k in fmt})
print(dict(st))
for r in ex[:3]: print(repr(r['text'][-220:]),'| gold',r['gold'],'pred',r['pred'])
```