Pass 47 has the C(−3.0) scores. In one block I'll compute the same paired table against the original (pass 46): Δlp and Δproj at onset and novel newlines for the new-loop and persistent groups. I'll also restore the submission C(−1.5) from /work/best_final and verify it against the saved arrays.

If the gain is multiplicative, the persistent-onset Δlp should be about twice C(−1.5)'s −0.147, while the new-loop Δlp should stay small.

```python
lp47=np.load('/work/fwd_47/token_logprobs.npy'); H47=np.load('/work/fwd_47/hidden_20.npy',mmap_mode='r')
assert (np.load('/work/fwd_47/input_ids.npy')==ids45).all()
rows=[]
for g in ['new','persist']:
    for kind,pos in [('onset',on),('novel',nov)]:
        per={}
        for i,q in pos:
            if meta45[i]['grp']!=g: continue
            per.setdefault(i,[]).append((float(lp47[i,q]-lp46[i,q]), float((H47[i,q].astype(np.float32)-H46[i,q].astype(np.float32))@dclean), float(lp45[i,q]-lp46[i,q])))
        m=np.array([np.mean(v,0) for v in per.values()]); se=m.std(0)/np.sqrt(len(m))
        rows.append(dict(grp=g,kind=kind,n=len(m),dlp_C3=round(m[:,0].mean(),3),se=round(se[0],3),dlp_C15=round(m[:,2].mean(),3),dproj_C3_sd=round(m[:,1].mean()/sd,2)))
# restore submission
for n in list(edited()): drop_edit(n)
fs=sorted(glob.glob('/work/best_final/*.npy'))
for p in fs: save_edit(os.path.basename(p)[:-4], torch.tensor(np.load(p),dtype=torch.float32))
ok=all(np.array_equal(load(os.path.basename(p)[:-4]).to(torch.bfloat16).float().numpy(), torch.tensor(np.load(p)).to(torch.bfloat16).float().numpy()) for p in fs)
with open('/work/notes.md','a') as f: f.write("After pass47: C(-1.5) restored from /work/best_final (submission).\n")
(pd.DataFrame(rows).to_string(), len(edited()), ok)
```