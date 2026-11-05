This is a real step forward. At h18, on held-out answers, every cut answer has a token above the 99.9th percentile of finished answers, and even the 10th percentile of the per-answer maximum (790) clears that level (677). At h12, newlines separate with d′ 2.47 and 95% of cut answers have a newline above the finished q999. So the late state is much cleaner at shallow depth.

The weakness is sample size: the "finished" control here is only about 20 answers, or 177 newlines, and a false alarm on a good answer is what would actually cost points. `pass15` and `pass11` hold many more finished answers with h12 and h18 recorded, so I can estimate the false-alarm tail properly at no pass cost. I'll fix the direction from all of `pass29`, then measure the share of finished answers, right and wrong, whose maximum crosses each threshold. For cut answers I'll measure the share crossing and the first crossing position. If some threshold passes about 90% of cut answers while keeping false alarms under about 1% of finished answers, the circuit is feasible. I'll also read which exit forms worked best in pass 10.

```python
dirs={}
for l,tag in [(12,'newline'),(12,'all'),(18,'all'),(18,'newline')]:
    h=np.load(FW+f'hidden_{l}.npy',mmap_mode='r')
    tm=nlm if tag=='newline' else valid
    d=np.asarray(h[late&tm]).astype(np.float32).mean(0)-np.asarray(h[early&tm]).astype(np.float32).mean(0); d[1038]=0; d/=np.linalg.norm(d)
    dirs[(l,tag)]=d
np.savez('/work/late_dirs.npz',**{f'{l}_{t}':v for (l,t),v in dirs.items()})
print('cos 18all vs u18,v18',round(float(dirs[(18,'all')]@z['u18']/np.linalg.norm(z['u18'])),2),round(float(dirs[(18,'all')]@z['v18']/np.linalg.norm(z['v18'])),2),' 12all vs u12,v12',round(float(dirs[(12,'all')]@z['u12']/np.linalg.norm(z['u12'])),2),round(float(dirs[(12,'all')]@z['v12']/np.linalg.norm(z['v12'])),2))
def ctrl_answers(fw):
    mm=json.load(open(fw+'meta.json')); idv=np.load(fw+'input_ids.npy',mmap_mode='r')
    return mm,idv
out={}
for fw in ['/work/fwd_15/','/work/fwd_11/']:
    mm,idv=ctrl_answers(fw); ln=np.array(mm['lengths']); nn=len(ln)
    print(fw,'texts',nn,'files',[f for f in os.listdir(fw) if 'hidden' in f])
    for (l,tag),d in dirs.items():
        h=np.load(fw+f'hidden_{l}.npy',mmap_mode='r'); mx=[]
        for i in range(nn):
            row=np.asarray(idv[i]); w=np.where(row==105)[0]
            if len(w)<1: continue
            s=int(w[-1])+3; e=min(ln[i],s+300)
            if e<=s+3: continue
            tm=np.arange(s,e)
            if tag=='newline':
                tm=tm[(row[tm]==108)|(row[tm]==107)]
                if len(tm)==0: continue
            mx.append(float((np.asarray(h[i,tm]).astype(np.float32)@d).max()))
        out[(fw,l,tag)]=np.array(mx)
print('finished control answers per set',{k[0][-7:]+str(k[1:]):len(v) for k,v in out.items()})
for (l,tag),d in dirs.items():
    h=np.load(FW+f'hidden_{l}.npy',mmap_mode='r'); pm=[];fp=[]
    tm_all=nlm if tag=='newline' else valid
    C=np.concatenate([out[(fw,l,tag)] for fw in ['/work/fwd_15/','/work/fwd_11/']])
    for i in np.where(iscut)[0]:
        mk=tm_all[i]&ansm[i]
        P=np.asarray(h[i][mk]).astype(np.float32)@d; pp=pos[i][mk]
        pm.append(float(P[pp>=380].max()) if (pp>=380).any() else -1e9)
    pm=np.array(pm)
    Ts=np.quantile(C,[.9,.99,.995,.999,1.0])
    print((l,tag),'ctrl answers',len(C),'answer-max q90/q99/q995/q999/max',np.round(Ts,0).tolist())
    print('   share cut answers past T:',[round(float((pm>t).mean()),2) for t in Ts],' by kind loop/list at q995:',round(float((pm[:30]>Ts[2]).mean()),2),round(float((pm[30:]>Ts[2]).mean()),2))
tm10=json.load(open('/work/tf10_meta.json')); print(type(tm10), (list(tm10.keys())[:8] if isinstance(tm10,dict) else tm10[0]))
```