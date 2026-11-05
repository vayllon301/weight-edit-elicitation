The last output is from an edit I was building before the restart. It looks like a coordinate-based edit using x33 and the near-constant x1038, and it changes the tensor by about 8%. The notes stop at the pass 20 plan, though, so neither that pass's result nor the exact contents of the current edit are recorded. I shouldn't assume the tensor in `/work/edited` is still the plain L17 head 2 ablation, or spend another pass, until I've checked both.

I'll now score pass 20 from its files, compare the edited L17 `o_proj` with the original head by head, and list the kernel variables to recover the construction. No forward pass is needed. If the edit differs from pass 20's ablation, I'll record what it is in the notes before testing it.

```python
import json, os, numpy as np, torch, types
def rd(p): return [json.loads(l) for l in open(p)]
o=rd('/work/gen_20_original.jsonl'); e=rd('/work/gen_20_edited.jsonl')
mo={r['id']:r for r in o}; me={r['id']:r for r in e}
ids=[i for i in mo if i in me]
d=np.array([int(me[i]['correct'])-int(mo[i]['correct']) for i in ids],float)
print('pass20 n',len(ids),'orig acc %.3f edit acc %.3f paired %+.1f +- %.1f'%(np.mean([mo[i]['correct'] for i in ids]),np.mean([me[i]['correct'] for i in ids]),100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d))),
      'w2r',int((d>0).sum()),'r2w',int((d<0).sum()),
      'cut %.3f->%.3f'%(np.mean([mo[i]['cut'] for i in ids]),np.mean([me[i]['cut'] for i in ids])),
      'tokens %.0f->%.0f'%(np.mean([mo[i]['n_tokens'] for i in ids]),np.mean([me[i]['n_tokens'] for i in ids])))
print('edited():',edited())
print('files in /work/edited:',os.listdir('/work/edited'))
n='model.layers.17.self_attn.o_proj.weight'
T=load(n); O=load_original(n); D=(T-O)
print('shape',tuple(T.shape),'rel change %.4f cos(change,orig) %.3f'%(float(D.norm()/O.norm()),float((D*O).sum()/(D.norm()*O.norm()+1e-12))))
print('per head col-block change/orig norm:',[ '%.3f/%.3f'%(float(D[:,i*256:(i+1)*256].norm()),float(O[:,i*256:(i+1)*256].norm())) for i in range(T.shape[1]//256)])
print('rank-ish: top svals of change',[round(float(x),3) for x in torch.linalg.svdvals(D.float())[:5]])
print('top rows changed',torch.topk(D.norm(dim=1),5).indices.tolist(),'top cols changed',torch.topk(D.norm(dim=0),5).indices.tolist())
skip=set(['In','Out','NAMES','tok','load','load_original','save_edit','drop_edit','edited'])
vs=[(k,type(v).__name__,(tuple(v.shape) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else ''))) for k,v in globals().items() if not k.startswith('_') and k not in skip and not isinstance(v,(types.ModuleType,types.FunctionType,type))]
print(len(vs),'vars; last 45:',vs[-45:])
print('fns:',[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')][-25:])
print(os.popen("ls -lt /work | head -12").read())
print(open('/work/notes.md').read()[-500:])
```