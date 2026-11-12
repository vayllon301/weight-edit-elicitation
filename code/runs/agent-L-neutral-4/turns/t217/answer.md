The opposite-sign test was already run in pass 11 on the same screen texts. Removing dclean (β = +2) moved onset probability the wrong way, from 0.73 to 0.78, while amplifying it (pass 12, β = −1.5) brought it to 0.60. The repetition-resisting role of the signal is therefore causally supported at two signs and several strengths, and no further pass is needed for it.

Two last questions can be answered from data already returned, without a pass.
- **Weak signal in both models?** Is the new loops' weak onset signal already there in the original model? That shows up as the onset gap computed on pass-46 (original) states.
- **How close was the divergence?** Find where each new-loop answer first diverges from the original's own answer. At that token, pass 46 gives the log-prob the original assigned to the edited model's token. Values near log(0.5) mean C only tipped a near-tie, the expected signature of a small, diffuse perturbation.

```python
H46n=H46
rows=[]
nprj46=np.array([float(H46n[i,q].astype(np.float32)@dclean) for i,q in nov])
mu46={t:nprj46[nf==t].mean() for t in set(nf) if (nf==t).sum()>=10}
nc46=np.array([p-mu46[t] if t in mu46 else np.nan for p,t in zip(nprj46,nf)]); sd46=np.nanstd(nc46)
for g in ['new','persist']:
    oo=[x for x in on if meta45[x[0]]['grp']==g and first(x) in mu46]
    op=np.array([float(H46n[i,q].astype(np.float32)@dclean)-mu46[first((i,q))] for i,q in oo])
    ng=np.array([c for c,(i,q) in zip(nc46,nov) if meta45[i]['grp']==g and not np.isnan(c)])
    rows.append(dict(grp=g,orig_onset_gap_sd=round((op.mean()-ng.mean())/sd46,2),orig_nov_mean_sd=round(ng.mean()/sd46,2)))
# divergence point for new loops vs original's own answer in the same pass
gen_orig={}
for k in passes:
    for l in open(f'/work/gen_{k}_original.jsonl'):
        r=json.loads(l); gen_orig[(k,r['id'])]=r['text']
dv=[]
for i,m in enumerate(meta45):
    if m['grp']!='new': continue
    p0=plen45[i]; a=ids45[i,p0+1:lens45[i]]
    ob=tok.encode(fmt.format(problem=prac[m['id']]['problem'])+gen_orig[(m['src'],m['id'])]).ids[p0+1:]
    d=next((j for j in range(min(len(a),len(ob))) if a[j]!=ob[j]),None)
    if d is None: continue
    q=p0+d
    dv.append((d, float(lp46[i,q]), float(lp45[i,q])))
dv=np.array(dv)
(rows, dict(n=len(dv),med_div_tok=float(np.median(dv[:,0])),orig_lp_of_edit_tok_med=round(float(np.median(dv[:,1])),2),
            frac_orig_p_gt_0p3=round(float((np.exp(dv[:,1])>0.3).mean()),2),edit_lp_med=round(float(np.median(dv[:,2])),2)))
```