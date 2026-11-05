The boost refutes the capacity hypothesis. Doubling L17 h0 moves copy digits by only +0.3 points, and it slightly hurts result digits (about -0.7), other-first digits, and retention of solved answers. So this head is necessary but not limiting: when it's present, reading numbers from the problem is already saturated. Then the 77% first-digit agreement must reflect the decision about which number to use, not failure to read it, and I'm dropping this as a repair target. The ablation and boost passes still have value for another question, so I'll get more from them without spending new passes.

I can run a loop test on the existing data. The 145 cut-off answers were in those same texts, so I can check whether ablating or boosting L17 h0 flips tokens inside repeated cycles more than it does in normal text. A clearly higher break rate in loops would implicate retrieval in loop maintenance. Meanwhile, I'll localise the reasoning failures by matching each reference's intermediate results against the model's text. That shows whether failures occur early (misreading) or late (losing track), and whether wrong final answers equal an intermediate reference value (stopping too soon) or an extra step past it. Any of these would be a concrete failure type to target.

```python
drop_edit('model.layers.17.self_attn.o_proj.weight'); print(edited())
open('/work/notes.md','a').write("pass5 result: L17h0 x2: copy +0.3pt only, result -0.7, own solved retention 0.990 => head necessary but NOT limiting; dropped. No edits active.\n")
# loop-break analysis using passes 4,5
lp_tk={k_:np.load(f'/work/fwd_{k_}/topk_ids.npy',mmap_mode='r')[:,:,0] for k_ in (4,5)}
idx_of={x['id']:n for n,x in enumerate(g)}
stats={4:[],5:[]}; flip_loop={4:[],5:[]}
for _,r_ in LD[LD.period.notna()].iterrows():
    i=idx_of[r_.id]; a=ids_np[i,:lens[i]]; s=ans_start(a); p_=int(r_.period); st_=int(r_.start)+s
    c2=np.arange(st_+p_,min(st_+2*p_,len(a)))
    tail=np.arange(st_+p_,len(a))
    for k_ in (4,5):
        f_=np.array(lp_tk[k_][i,tail-1])!=a[tail]
        flip_loop[k_].append(f_.mean())
        stats[k_].append((np.array(lp_tk[k_][i,c2-1])!=a[c2]).any())
for k_ in (4,5):
    print(k_,'loop token flip rate %.4f; cycle2 has >=1 flip %.3f'%(np.mean(flip_loop[k_]),np.mean(stats[k_])))
# failure depth
rows5=[]
for x in g:
    if x['cut']: pass
    sol=pid[x['id']]['solution']
    anns=[m.rsplit('=',1)[1].strip() for m in re.findall(r'<<([^>]*)>>',sol) if '=' in m]
    vals=[]
    for v in anns:
        try: vals.append(float(v.replace(',','').replace('$','').rstrip('.')))
        except: pass
    tn=[]
    for m in re.findall(r'\d[\d,]*\.?\d*',x['text']):
        try: tn.append(f(m))
        except: pass
    found=[any(abs(v-t)<1e-6 for t in tn) for v in vals]
    fm=next((i for i,b in enumerate(found) if not b),None)
    try: pv=float(str(x['pred']).replace(',',''))
    except: pv=None
    rows5.append(dict(id=x['id'],n=len(vals),correct=x['correct'],cut=x['cut'],first_miss=fm,frac=np.mean(found) if found else np.nan,
      pred_is_intermediate=(pv is not None and any(abs(pv-v)<1e-6 for v in vals[:-1])),
      gold_in_text=any(abs(float(str(x['gold']).replace(',',''))-t)<1e-6 for t in tn)))
F=pd.DataFrame(rows5); W=F[(~F.correct)&(~F.cut)&(F.n>=2)]
print('wrong finished n',len(W),'pred=earlier ref intermediate %.3f'%W.pred_is_intermediate.mean(),'gold anywhere in text %.3f'%W.gold_in_text.mean(),'frac intermediates found %.3f'%W.frac.mean())
print('first missing step index',W.first_miss.fillna(-1).astype(int).value_counts().sort_index().to_dict())
print('as share of steps',(W.first_miss.fillna(W.n)/W.n).round(1).value_counts().sort_index().to_dict())
C_=F[(F.cut)&(F.n>=2)]; print('cut frac intermediates %.3f'%C_.frac.mean(),'first miss',C_.first_miss.fillna(-1).astype(int).value_counts().sort_index().to_dict())
```