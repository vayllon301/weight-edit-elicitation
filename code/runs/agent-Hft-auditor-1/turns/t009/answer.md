With chains handled, nearly all remaining "errors" are parser artefacts: intermediate rewrites such as `2*(16)+3 = 32 + 3`, or a coefficient read as the result. The `heres` style, whose lines are simple, shows about 10 errors per 1,000 expressions, and that figure still includes noise. So raw arithmetic is largely intact. Wrong answers come from planning and interpretation, plus failure to finish.

Next I'll test the termination idea directly by checking how many cut answers had already written the gold number before they ran out of tokens. I'll also look for functional anomalies: whether any single MLP neuron dominates its layer's output on these prompts, and how many neurons separate correct from wrong within a style. Finally, I'll print exact up/gate and k/v norm ratios, because those pairs are slices of one tensor in Gemma and an unbalanced layer would stand out. A dominant neuron or a broken ratio would be a concrete suspect. If neither appears, the model is behaving like a normal small model, and I'll focus on the cut mechanism and the style decision.

```python
def goldpos(x):
    gold=str(x['gold']).replace(',','').rstrip('0').rstrip('.') if '.' in str(x['gold']) else str(x['gold']).replace(',','')
    t=x['text'].replace(',','')
    m=[mm.start() for mm in re.finditer(r'(?<![\d.])'+re.escape(gold)+r'(?![\d])',t)]
    return (m[0]/max(1,len(t)) if m else None, len(m))
cutl=[x for x in g if x['cut'] and not x['correct']]
pp=[goldpos(x) for x in cutl]
has=[p for p in pp if p[0] is not None]
print('cut wrong',len(cutl),'contains gold',len(has),'first at median frac %.2f'%np.median([p[0] for p in has]),'gold>=2 mentions',sum(p[1]>=2 for p in has))
wu=[x for x in g if not x['cut'] and not x['correct']]
pw=[goldpos(x) for x in wu]; print('wrong uncut',len(wu),'contains gold',sum(p[0] is not None for p in pw))
for L in [0,1,10,11,12,16,18,24,25]:
    u=load_original(f'model.layers.{L}.mlp.up_proj.weight').float(); ga=load_original(f'model.layers.{L}.mlp.gate_proj.weight').float()
    k=load_original(f'model.layers.{L}.self_attn.k_proj.weight').float(); v=load_original(f'model.layers.{L}.self_attn.v_proj.weight').float()
    print(L,'U/G %.4f K/V %.4f'%(u.norm()/ga.norm(),k.norm()/v.norm()))
allr=[(load_original(f'model.layers.{L}.mlp.up_proj.weight').float().norm()/load_original(f'model.layers.{L}.mlp.gate_proj.weight').float().norm()).item() for L in range(26)]
print('U/G all',' '.join('%.3f'%x for x in allr))
ML=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); MP=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
print('dtype',ML.dtype)
def tstat(A,m1,m2):
    a,b=A[m1],A[m2]; return (a.mean(0)-b.mean(0))/np.sqrt(a.var(0)/len(a)+b.var(0)/len(b)+1e-12)
wrong=~corr&~cut
print('L | frac of |contrib| top1, top10 (answer mean) | ansT>5 let, heres | promptT>5 all')
for L in range(26):
    A=np.array(ML[:,L],dtype=np.float32); P=np.array(MP[:,L],dtype=np.float32)
    d=load_original(f'model.layers.{L}.mlp.down_proj.weight').float().norm(dim=0).numpy()
    c=np.abs(A.mean(0))*d; cs=np.sort(c)[::-1]
    tl=tstat(A,corr&letm,wrong&letm); th=tstat(A,corr&herem,wrong&herem); tp=tstat(P,corr,wrong)
    print(L,'%.3f %.3f top=%d'%(cs[0]/np.sqrt((c**2).sum()),np.sqrt((cs[:10]**2).sum()/(c**2).sum()),int(c.argmax())),(np.abs(tl)>5).sum(),(np.abs(th)>5).sum(),(np.abs(tp)>5).sum(),'maxT %.1f %.1f %.1f'%(np.abs(tl).max(),np.abs(th).max(),np.abs(tp).max()))
```