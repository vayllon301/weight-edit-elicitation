The tool's input_ids drop the duplicated BOS that `tok.encode` adds, since the prompt template already contains one. My earlier `plen45` values are still correct in the tool's indexing; only the original-answer tokens in the divergence calculation were shifted by one. Re-running just that part with the extra BOS removed:

```python
dv=[]
for i,m in enumerate(meta45):
    if m['grp']!='new': continue
    p0=plen45[i]; a=ids45[i,p0+1:lens45[i]]
    ob=tok.encode(fmt.format(problem=prac[m['id']]['problem'])+gen_orig[(m['src'],m['id'])]).ids[1:][p0+1:]
    d=next((j for j in range(min(len(a),len(ob))) if a[j]!=ob[j]),None)
    if d is None: continue
    q=p0+d; dv.append((d, float(lp46[i,q]), float(lp45[i,q]), len(a)))
dv=np.array(dv)
dict(n=len(dv),med_div_tok=float(np.median(dv[:,0])),q25_75=(float(np.percentile(dv[:,0],25)),float(np.percentile(dv[:,0],75))),
     orig_p_of_edit_tok_med=round(float(np.exp(np.median(dv[:,1]))),2),frac_orig_p_gt_0p3=round(float((np.exp(dv[:,1])>0.3).mean()),2),
     edit_p_med=round(float(np.exp(np.median(dv[:,2]))),2))
```