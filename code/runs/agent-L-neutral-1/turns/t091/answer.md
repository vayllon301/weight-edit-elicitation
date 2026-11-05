The stop line is now closed, this time on direct evidence. Right after the model reaches gold, the median smallest gap to `####` is 5.4 nats, against 7.9 at legitimate intermediate boundaries. Shifting the threshold by 5 nats would convert 7 of 18 overruns and truncate 7 of 99 correct answers. That's no net gain, and the 2–3 nat shifts catch only 3 overruns, so a stronger "finish" circuit has no clean signal to amplify. One caveat: I only scored `####`, not the "Final Answer" openings, but those are a small minority of endings. The five last-step errors I read are ordinary reasoning mistakes, such as dividing by 4 instead of 2 or double-counting a term, and none is a repeated defect.

The strongest evidence left is that accuracy is sensitive to some directions: the +1 writer cost 8.6 points. So I want a direction with difficulty removed by design. I have 176 cases of the same problem answered right and wrong, and that lets me do what my earlier contrasts couldn't. I'll build a teacher-forced set of those pairs, excluding cut answers so loops don't dominate, and truncating for size. I'll also balance by which model produced the right answer, so that style cancels. If residual or neuron differences are consistent across pairs, with paired d well above the shuffled null, that's a candidate for a causal test. If they sit at the null, no accuracy direction is visible at this resolution and I won't spend generation passes on it.

```python
import json, re, numpy as np, collections
note="""
Loop/stop reanalysis (no pass): cut rate is flat by steps (1-step .34). Strict overrun (gold as eq result, not in problem): cut 24=1.96pt, finished wrong 29=2.37pt. Run-on numbered lists >10 items: 111 answers (9%), all cut, acc 2-3%; lists <=6 items cut <2%. Non-list cut .30 (exact loops). Ceiling if cut answers finished at same-progress accuracy 4.45pt. Stop gap to #### after gold 5.4 nats vs 7.9 legit; 5-nat shift flips 7/18 overruns, truncates 7/99 correct => stop threshold line CLOSED, no separable signal. Spurious constants: 5.5% of finished, own novel operand rate below refs => CLOSED. First-divergence right vs wrong: word vs word at token ~24, no class enrichment => style-level chaos, not the error site.
"""
open('/work/notes.md','a').write(note)
print(prompt_of.__code__.co_varnames[:prompt_of.__code__.co_argcount])
print(open('/practice/eval_prompt_format.txt').read()[:260].replace('\n','\\n'))
q0=P[g1[0]['id']]
try:
    pr=prompt_of(q0['problem']); print('prompt_of(problem) prefix ok:',texts[0].startswith(pr))
except Exception as e:
    print('err',e)
ptxt=[]; seen={}
def add(t):
    if t not in seen:
        seen[t]=len(ptxt); ptxt.append(t)
    return seen[t]
MAXA=420
meta=[]
for c,na,nb,k,r,w,rn in pairs:
    if r['cut'] or w['cut']: continue
    pr=prompt_of(P[k]['problem'])
    def build(ans):
        ids=enc(ans)[1:] if enc(ans)[0]==2 else enc(ans)
        return pr+(tok.decode(ids[:MAXA],skip_special_tokens=False) if len(ids)>MAXA else ans)
    ir=add(build(r['text'])); iw=add(build(w['text']))
    meta.append(dict(c=c,pair=na+'-'+nb,id=k,right=ir,wrong=iw,rn=rn))
print('usable pairs',len(meta),'unique texts',len(ptxt))
print(collections.Counter((x['pair'],x['rn']) for x in meta))
lens=[len(enc(t)) for t in ptxt]; print('tokens max',max(lens),'median',int(np.median(lens)),'sum',sum(lens))
json.dump(ptxt,open('/work/tf15_texts.json','w')); json.dump(meta,open('/work/tf15_meta.json','w'))
gb=len(ptxt)*max(lens)*(1152*4+6912*3)*2/1e9; print('array GB est (4 hidden + 3 mlp)',round(gb,2))
```