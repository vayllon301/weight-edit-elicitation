I need to read this carefully, because the headline is misleading. Escape jumped from 4.9% to 65.9% (126 new, 1 lost), which looks like a loop fix. But per-token mismatch rose tenfold everywhere, from 0.29% to 2.97% on correct answers and to 4.19% on wrong ones. At about 3% per token, a 50-token repeat would show at least one mismatch roughly 78% of the time by chance alone. So a 66% "escape" may be nothing more than general perturbation. The logprob of copied tokens fell only 0.028, which also suggests copying is barely touched. My any-mismatch metric saturates, as I noted for the controls, and I shouldn't have compared it with per-token damage.

The fair comparison uses the same unit on both sides: per-token mismatch inside the first repeat, against per-token mismatch on correct and wrong answers. A selective loop edit should change copied tokens at a clearly higher rate than ordinary tokens. If the first-repeat rate is about the same or lower, E12 has no loop selectivity and is just a broad perturbation. I'll also compare both passes on the late-loop tokens, where the pointer is fully locked. No pass is needed for this.

```python
def per_token(dr):
    ids=np.load(dr+'input_ids.npy',mmap_mode='r'); tk=np.load(dr+'topk_ids.npy',mmap_mode='r')
    c=collections.Counter()
    for k,m in enumerate(M35):
        L_=int((np.asarray(ids[k])>=0).sum()); arr=np.asarray(ids[k,:L_]); top=np.asarray(tk[k,:L_,0]); mis=(top[:L_-1]!=arr[1:]); s0=m['ans_start']
        if m['kind']=='loop':
            a=s0+m['on']-1; b_=min(a+m['P'],L_-1)
            c['first_mis']+=int(mis[a:b_].sum()); c['first_n']+=b_-a
            c['late_mis']+=int(mis[b_:].sum()); c['late_n']+=max(0,L_-1-b_)
            c['pre_mis']+=int(mis[s0-1:a].sum()); c['pre_n']+=a-(s0-1)
        else:
            a=s0-1; key=m['kind']; c[key+'_mis']+=int(mis[a:].sum()); c[key+'_n']+=L_-1-a
    return {x:round(100*c[x+'_mis']/max(1,c[x+'_n']),2) for x in ['pre','first','late','correct','wrong']},{x:c[x+'_n'] for x in ['pre','first','late','correct','wrong']}
pb,ns=per_token('/work/fwd_35/'); pe,_=per_token('/work/fwd_36/')
print('tokens',ns); print('baseline %',pb); print('E12 %',pe)
print('ratio first/correct E12 %.2f, late/correct %.2f, pre-loop/correct %.2f'%(pe['first']/pe['correct'],pe['late']/pe['correct'],pe['pre']/pe['correct']))
```