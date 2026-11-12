On hold_3, B gives **+5.1 ± 2.4**, against +4.7 for E6. Cuts rose more than under E6 (32 to 48, against 32 to 37). The original's results match those in E6's passes, so the baselines line up.

Next I'll pool B over all four HOLDOUT sets and compare it with E6 problem by problem. The two were measured in separate passes against identical original baselines. I'll also compare cut counts, then restore E6 from best_4.

My expectation is that B's pooled gain is about +5.1, close to E6's +5.37. That would confirm the attribution: the regime neuron carries almost all of the HOLDOUT gain, and the doubt columns' contribution is within noise but slightly positive, with fewer cut-offs. As pre-registered, the submission stays E6.

```python
def pool(ks):
    d=[];ed={};cut=[0,0]
    for k in ks:
        o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
        for i in o:
            d.append(int(e[i]['correct'])-int(o[i]['correct'])); ed[i]=int(e[i]['correct']); cut[0]+=o[i]['cut']; cut[1]+=e[i]['cut']
    d=np.array(d); return d,ed,cut
dE,edE,cE=pool([28,29,30,31]); dB,edB,cB=pool([38,39,40,41])
ids_=sorted(edE); x=np.array([edE[i]-edB[i] for i in ids_])
out=dict(E6=(round(100*dE.mean(),2),round(100*dE.std(ddof=1)/np.sqrt(len(dE)),2),cE),B=(round(100*dB.mean(),2),round(100*dB.std(ddof=1)/np.sqrt(len(dB)),2),cB),
         E6_minus_B=(round(100*x.mean(),2),round(100*x.std(ddof=1)/np.sqrt(len(x)),2),int((x>0).sum()),int((x<0).sum())))
for f in os.listdir('/work/best_4'):
    save_edit(f[:-4],torch.from_numpy(np.load(f'/work/best_4/{f}').astype(np.float32)))
same=all(np.array_equal(np.load(f'/work/edited/{f}'),np.load(f'/work/best_4/{f}')) for f in os.listdir('/work/best_4')) and len(os.listdir('/work/edited'))==3
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 41 result (B) hold_3: +5.1+-2.4, cuts 32->48.\nATTRIBUTION HOLDOUT pooled: {out}\nE6 RESTORED from best_4 (verified={same}). Submission = E6.\n")
out, same, edited()
```