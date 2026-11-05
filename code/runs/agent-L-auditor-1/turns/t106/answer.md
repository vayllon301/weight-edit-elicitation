T4 reaches 58.6%, up from T3's 53.9%. Wrong answers fell from 196 to 155, though cut-offs rose from 40 to 57. So this windowed-layer cut does help accuracy, but something about the loop behaviour changed, and that needs a look: a rising cut-off rate could cost accuracy at the next step.

I'll compute the exact paired numbers against T3 and the original, checking whether the gain is consistent across subsets. I'll also look at what the new cut-offs are. If T4's cut answers came from problems that T3 got right, they'd be real losses to recover. If they came mostly from problems that T3 got wrong, they cost little. I'll save this as a checkpoint either way.

```python
import json, shutil, os, numpy as np
g4=rd('/work/gen_23.jsonl')
print('T4 vs orig %+.2f se %.2f w->r %d r->w %d'%paired(g0,g4,ids))
print('T4 vs T3   %+.2f se %.2f w->r %d r->w %d'%paired(g3,g4,ids))
print('fifths vs T3',[round(float(paired(g3,g4,ids[k::5])[0]),1) for k in range(5)])
c3=set(i for i in ids if g3[i]['cut']); c4=set(i for i in ids if g4[i]['cut'])
newc=[i for i in ids if i in c4 and i not in c3]; gone=[i for i in ids if i in c3 and i not in c4]
print('cut T3 %d T4 %d new %d gone %d; new-cut were correct in T3 %.3f ; gone-cut now correct %.3f'%(len(c3),len(c4),len(newc),len(gone),np.mean([g3[i]['correct'] for i in newc]) if newc else 0,np.mean([g4[i]['correct'] for i in gone]) if gone else 0))
print('T4 style',{s:(sum(style(g4[i]['text'])==s for i in ids),round(float(np.mean([g4[i]['correct'] for i in ids if style(g4[i]['text'])==s])),3)) for s in ('Let','Heres','other') if any(style(g4[i]['text'])==s for i in ids)})
print('marker',np.mean([g4[i]['used_marker'] for i in ids]),'toks',np.mean([g4[i]['n_tokens'] for i in ids]))
for lo,hi in bins:
    x=[i for i in ids if lo<=nsteps(i)<=hi]
    print('steps',lo,hi,'n',len(x),'T3 %.3f T4 %.3f cut %.3f'%(np.mean([g3[i]['correct'] for i in x]),np.mean([g4[i]['correct'] for i in x]),np.mean([g4[i]['cut'] for i in x])))
def tail_repeat(t):
    w=t[-200:]; return t.count(w[-60:])
rep=[tail_repeat(g4[i]['text']) for i in c4]; print('cut: tail 60-char repeats median',np.median(rep),'share >=3',np.mean(np.array(rep)>=3))
print('cut: prefix ok? last number equals gold anywhere in text share %.3f'%np.mean([str(g4[i]['gold']) in g4[i]['text'].replace(',','') for i in c4]))
rng=np.random.default_rng(5)
for i in list(rng.choice(sorted(c4),3,replace=False)):
    t=g4[i]['text'].encode('ascii','replace').decode().replace('\n',' / ')
    print('--CUT',i,'gold',g4[i]['gold'],'|',t[:140],'...',t[-140:])
os.makedirs('/work/best_7',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f,'/work/best_7/'+f)
open('/work/notes.md','a').write("\npass 23 result T4 (T3 + L10 h1 du10) dev_a acc .5859 cut .119, saved /work/best_7. T3 .5391 cut .084.\n")
```