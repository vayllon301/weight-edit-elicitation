The pass returned 335 texts with layer-20 states. Now comes the block-126 analysis, applied per group. First the onset, novel and in-loop newlines are found. Each hidden state at an onset or novel newline is centred on the mean of novel newlines sharing the same next token, pooled over all groups, so every group is measured from one common reference.

For each group I report three numbers:
- **gap_sd:** how far onset newlines sit above that group's own novel newlines along dclean, in SD units of the pooled novel lines.
- **mean onset log-prob:** the log-probability of the first token of the repeated line.
- **share above novel P90:** the share of onsets above the 90th percentile of novel lines.

If gap_sd for new loops is similar to the persistent loops', they share the onset channel.

```python
ids45=np.load('/work/fwd_45/input_ids.npy'); lp45=np.load('/work/fwd_45/token_logprobs.npy')
lens45=json.load(open('/work/fwd_45/meta.json'))['lengths']; meta45=json.load(open('/work/newloop_meta.json'))
plen45=[len(tok.encode(fmt.format(problem=prac[m['id']]['problem'])).ids)-1 for m in meta45]
on=[];nov=[]
for i in range(len(meta45)):
    a=ids45[i,:lens45[i]]; p0=plen45[i]
    nls=[q for q in range(p0,len(a)-1) if a[q] in NL]; seen=set(); found=False
    for j,q in enumerate(nls):
        end=nls[j+1] if j+1<len(nls) else len(a); line=tuple(a[q+1:end])
        if len(line)>=4 and line in seen:
            if not found: on.append((i,q)); found=True
        elif not found and len(line)>=4: nov.append((i,q))
        seen.add(line)
H=np.load('/work/fwd_45/hidden_20.npy',mmap_mode='r')
def first(iq): return int(ids45[iq[0],iq[1]+1])
nf=np.array([first(x) for x in nov]); nprj=np.array([float(H[i,q].astype(np.float32)@dclean) for i,q in nov])
mu={t:nprj[nf==t].mean() for t in set(nf) if (nf==t).sum()>=10}
nc=np.array([p-mu[t] if t in mu else np.nan for p,t in zip(nprj,nf)]); sd=np.nanstd(nc); p90=np.nanpercentile(nc,90)
rows=[]
for g in ['new','persist','fin']:
    gi=set(k for k,m in enumerate(meta45) if m['grp']==g)
    oo=[x for x in on if x[0] in gi and first(x) in mu]
    op=np.array([float(H[i,q].astype(np.float32)@dclean)-mu[first((i,q))] for i,q in oo])
    ng=np.array([c for c,(i,q) in zip(nc,nov) if i in gi and not np.isnan(c)])
    olp=np.array([lp45[i,q] for i,q in oo])
    rows.append(dict(grp=g,n_txt=len(gi),n_onset=sum(1 for x in on if x[0] in gi),n_on_used=len(oo),n_nov=len(ng),
        nov_mean_sd=round(ng.mean()/sd,2) if len(ng) else None,onset_gap_sd=round((op.mean()-ng.mean())/sd,2) if len(oo) else None,
        onset_lp=round(float(olp.mean()),2) if len(oo) else None,frac_above_p90=round(float((op>p90).mean()),2) if len(oo) else None))
pd.DataFrame(rows).to_string()
```