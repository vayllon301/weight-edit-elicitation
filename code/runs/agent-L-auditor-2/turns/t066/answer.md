Accuracy fell by 3.9 ± 2.2 points and cut-offs rose, which goes against my prediction. Before I interpret that, I need to know what the edit actually did. If few problems converted, this says little about the mode. If most converted and accuracy fell, L-mode isn't causally better, and its advantage in the baseline was selection.

So I'll measure three things:

- the real write size at the last prompt token
- the style transitions
- paired accuracy, separately for problems that changed style and problems that didn't

The last split is the key one. If the damage falls on problems that kept their style, the edit is hurting through a side effect. If it falls on converted problems, the mode hypothesis is wrong.

```python
import numpy as np, json
from collections import Counter
o,e=load_gen2(11)
so=np.array([sty(x['text']) for x in o]); se=np.array([sty(x['text']) for x in e])
print('orig',Counter(so),'edited',Counter(se),'transitions',Counter(a+b for a,b in zip(so,se)))
co=np.array([x['correct'] for x in o],float); ce=np.array([x['correct'] for x in e],float); cuto=np.array([x['cut'] for x in o],float); cute=np.array([x['cut'] for x in e],float)
for lab,m in [('H->L',(so=='H')&(se=='L')),('H->H',(so=='H')&(se=='H')),('L->L',(so=='L')&(se=='L')),('L->H',(so=='L')&(se=='H'))]:
    if m.sum()==0: continue
    d=ce[m]-co[m]
    print(lab,int(m.sum()),'acc',round(co[m].mean(),3),'->',round(ce[m].mean(),3),'diff',round(d.mean(),3),'se',round(d.std()/np.sqrt(m.sum()),3),'cut',round(cuto[m].mean(),3),'->',round(cute[m].mean(),3),'identical text',sum(x['text']==y['text'] for x,y,mm in zip(o,e,m) if mm))
print('identical overall',sum(x['text']==y['text'] for x,y in zip(o,e)))
tv=np.load('/work/modeLH_h16.npy'); u=tv/np.linalg.norm(tv)
Ho=np.load('/work/gen_11_original_acts/hidden_last_prompt.npy',mmap_mode='r'); He=np.load('/work/gen_11_edited_acts/hidden_last_prompt.npy',mmap_mode='r')
# layers order [13,16,20,26]
for i,l in enumerate([13,16,20,26]):
    d=np.asarray(He[:,i],dtype=np.float32)-np.asarray(Ho[:,i],dtype=np.float32); md=d.mean(0)
    print('hid',l,'mean shift norm',round(float(np.linalg.norm(md)),1),'sd over items of norm',round(float(np.linalg.norm(d,axis=1).std()),1),'cos with tvec16' if l==16 else '',round(float(md@u/np.linalg.norm(md)),3) if l==16 else '','proj' if l==16 else '',round(float(md@u),1) if l==16 else '')
# margin along u at h16, original
X=np.asarray(Ho[:,1],dtype=np.float32); p=(X-X.mean(0))@u
print('orig h16 proj: L mean',round(float(p[so=='L'].mean()),1),'H mean',round(float(p[so=='H'].mean()),1),'sd L',round(float(p[so=='L'].std()),1),'sd H',round(float(p[so=='H'].std()),1))
print('H problems proj: converted',round(float(p[(so=='H')&(se=='L')].mean()),1) if ((so=='H')&(se=='L')).any() else None,'not converted',round(float(p[(so=='H')&(se=='H')].mean()),1))
hh=np.where((so=='H')&(se=='H'))[0][:3]
for i in hh:
    print('\nH stayed: first differing at char',next((k for k,(a,b) in enumerate(zip(o[i]['text'],e[i]['text'])) if a!=b),None)); print(' o:',o[i]['text'][:120].replace('\n',' | ')); print(' e:',e[i]['text'][:120].replace('\n',' | '))
```