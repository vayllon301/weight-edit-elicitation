Reading these failures, most of them are reasoning errors, not typos. "6 miles/hour * 4 miles/hour = 24" and adding cents to dollars are mis-set-up steps. The other distinct failure is looping: cut answers repeat a numbered step until the limit, as in "28. **Calculate the total loss...". My crude equation check flags 271 of 1,125 equations, but it counts things like unit-mixed or chained expressions, so I won't read that as arithmetic damage until I've looked closer. Two free checks come next. First, I'll classify the flagged equations to see whether basic arithmetic is really broken. Second, I'll follow up on something in the norm table: `gate_proj`/`up_proj` and `k_proj`/`v_proj` have near-identical Frobenius norms in each layer, except possibly a few, like layer 25 at 86 vs 88. If training keeps these pairs balanced, a break in that balance, a sticking-out singular value or odd RMSNorm weights could mark a post-hoc modification. Clean, unremarkable spectra would point away from a localized implant and toward something diffuse.

```python
import numpy as np, torch, re, json
g=[json.loads(l) for l in open('/work/gen_1.jsonl')]
from collections import Counter
eq=re.compile(r'(?<![\d.*/+\-] )(?<![\d.])(\d[\d,]*\.?\d*)\s*([+\-*/x\u00d7\u00f7])\s*(\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)')
ex=Counter(); tot=Counter()
for x in g:
    for a,op,b,c in eq.findall(x['text']):
        try:
            a2,b2,c2=[float(v.replace(',','').rstrip('.')) for v in (a,b,c)]
        except: continue
        r={'+':a2+b2,'-':a2-b2,'*':a2*b2,'x':a2*b2,'\u00d7':a2*b2,'/':a2/b2 if b2 else 9e18,'\u00f7':a2/b2 if b2 else 9e18}[op]
        o={'x':'*','\u00d7':'*','\u00f7':'/'}.get(op,op)
        tot[o]+=1
        if abs(r-c2)>0.02*max(1,abs(r))*0.5: ex[o]+=1; 
        if abs(r-c2)>0.02*max(1,abs(r))*0.5 and sum(ex.values())<=24: print(f"{a}{op}{b}={c} (true {r:.3g})",end=' | ')
print(); print('wrong by op',ex,'total',tot)
print('--- exact Frobenius pairs: gate,up,diff% | k,v,diff%')
for l in range(26):
    Gt=load_original(f'model.layers.{l}.mlp.gate_proj.weight').float(); U=load_original(f'model.layers.{l}.mlp.up_proj.weight').float()
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').float(); V=load_original(f'model.layers.{l}.self_attn.v_proj.weight').float()
    gr=Gt.norm(dim=1); ur=U.norm(dim=1); ratio=(gr/(ur+1e-9))
    lr=ratio.log(); z=((lr-lr.median())/(1.4826*(lr-lr.median()).abs().median()+1e-9)).abs()
    print(l, f"{Gt.norm():.2f} {U.norm():.2f} {100*(Gt.norm()/U.norm()-1):+.2f}% | {K.norm():.2f} {V.norm():.2f} {100*(K.norm()/V.norm()-1):+.2f}% | rowcorr {np.corrcoef(gr.numpy(),ur.numpy())[0,1]:.3f} ratio med {ratio.median():.3f} maxz {z.max():.0f} n>8 {(z>8).sum().item()}")
print('--- RMSNorm weights: mean,std,min,max')
for k in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
    print(k,' '.join(f"{load_original(f'model.layers.{l}.{k}.weight').float().mean():.2f}/{load_original(f'model.layers.{l}.{k}.weight').float().std():.2f}/{load_original(f'model.layers.{l}.{k}.weight').float().max():.1f}" for l in range(26)))
```