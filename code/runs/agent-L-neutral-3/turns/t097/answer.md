Now the calibration table. For each opener I'll compute its mean first-token logprob and its gap to the best natural opener across the 102 prompts, and print how the top continuation token splits, using a fixed list of ordinary words only. As a reference for reachability, We needed a gap of about -9.5 at the strength that won on 94% of prompts, and that strength is near the saturation ceiling. An opener at a similar or smaller gap is reachable with the existing write. One with a gap beyond about -12 would need a second write site.

```python
tp22=np.load('/work/fwd_22/token_logprobs.npy',mmap_mode='r'); ti22=np.load('/work/fwd_22/topk_ids.npy',mmap_mode='r'); tl22=np.load('/work/fwd_22/topk_logprobs.npy',mmap_mode='r')
m22=json.load(open('/work/fwd_22/meta.json')); L22=np.array(m22['lengths'])
tm22=json.load(open('/work/tf22_meta.json'))
vals=np.array([float(tp22[i,L22[i]-2]) for i in range(len(L22))])
opsv=np.array([x['op'] for x in tm22]); piv=np.array([x['pi'] for x in tm22])
bestnat=np.maximum(vL13,vH13)
ok_wl=[',',':','.',' find',' we',' let',' step',' solve',' ',' 1','\n',"'","'s",' the',' total',' Let',' First',' problem',' calculate',' Step',' Solution','1','2',' to',' is',' are',' number',' We',' The',' Calculate',' Understand',' Determine',' Find',' identify',' determine',' understand','Step','Solution','Problem',' need',' Problem',' How',' how']
okd={}
for s in ok_wl:
    e_=E(s)
    if len(e_)==1: okd[e_[0]]=s
print('op      meanlp  sd   gap-to-best q05..q95 | next token top1 (share)')
for op in ops:
    m_=opsv==op; v=vals[m_]; gp_=v-bestnat[piv[m_]]
    rowi=np.where(m_)[0]; nxt=[int(ti22[i,L22[i]-1,0]) for i in rowi]
    c_=Counter(okd.get(t,'OTHER') for t in nxt).most_common(4)
    print(f'{op:9s}{v.mean():7.2f}{v.std():5.2f}{gp_.mean():8.2f} [{np.quantile(gp_,.05):6.2f},{np.quantile(gp_,.95):6.2f}] | {[(repr(k),n) for k,n in c_]}')
print('natural best mean on these prompts',float(bestnat[:102].mean()))
with open('/work/notes.md','a') as f:
    f.write('\nPass 22 screen done (first-token logprobs of 10 openers on 102 prompts), table in run output.\n')
```