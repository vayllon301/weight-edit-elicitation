The pooled result is clear. **Q11+L2 gives +12.7 ± 2.1** (+13.3 and +12.1), which isn't better than Q11 alone at +14.3, though cut-offs fell from about .17 to .08. So my prediction failed. Those extra finishers didn't add correct answers, and reasoning quality matters more than finishing rate. This is the warning from the task's method notes: judge by accuracy, not by fewer cut-offs. L2 was useful for offsetting L17's damage inside P1, but with L17 gone it has no job left, so **I'm dropping L2**. The simpler single-tensor edit is better on evidence and on simplicity.

To be precise rather than just comparing headlines, I'll compare the Q11 and Q11+L2 arms directly and see what happened to the answers L2 rescued. I also still need the dose-response required by the method, and the L11 sink direction has a useful property to check first. The projection subtracts a term proportional to each key's overlap with that direction. So strength above 1 would actively penalize sink-like keys and favor keys on the opposite side. Before spending passes, I'll measure where each token class sits on that axis using existing L11 hidden states, which is an allowed summary check. If digits sit on the negative side, strength above 1 should push attention further toward numbers, and it's worth testing.

```python
print(inspect.getsource(park_q)[:1100])
def arm(p): return {r['id']:r for r in lg(f'/work/gen_{p}_edited.jsonl')}
pairs=[(28,32),(29,33)]; bb=[]
for a_,b_ in pairs:
    A_=arm(a_);B_x=arm(b_)
    bb+=[(A_[k],B_x[k]) for k in A_]
n=len(bb); d=np.array([int(y['correct'])-int(x['correct']) for x,y in bb],float)
print('Q11L2 minus Q11: %.2f +- %.2f  L2 gains %d, L2 losses %d'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(n),(d>0).sum(),(d<0).sum()))
esc=[(x,y) for x,y in bb if x['cut'] and not y['cut']]; print('Q11 cut -> Q11L2 finished',len(esc),'right',sum(y['correct'] for x,y in esc))
new=[(x,y) for x,y in bb if not x['cut'] and y['cut']]; print('Q11 fin -> Q11L2 cut',len(new),'lost',sum(x['correct'] for x,y in new))
ff=[(x,y) for x,y in bb if not x['cut'] and not y['cut']]; print('both fin n',len(ff),'acc %.3f -> %.3f'%(np.mean([x['correct'] for x,y in ff]),np.mean([y['correct'] for x,y in ff])))
print('median tokens fin both: %.0f -> %.0f'%(np.median([x['n_tokens'] for x,y in ff]),np.median([y['n_tokens'] for x,y in ff])))
# where do token classes sit on the L11 sink axis (summary check, no network simulation)
print(os.listdir('/work/fwd_4')[:6], 'hidden_11' in ' '.join(os.listdir('/work/fwd_4')))
```