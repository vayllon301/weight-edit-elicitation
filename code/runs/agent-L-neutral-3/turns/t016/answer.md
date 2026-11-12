The pass worked. First I'll check the mechanics: how the tool tokenised the text (one `<bos>` or two), and whether teacher-forced top-1 predictions reproduce the greedy answers. If they don't, the token-level data wouldn't describe the states seen in generation. Then I'll extract the first-token distribution for each problem and verify token-level selectivity of the neuron candidates at layers 18 and 23. For each neuron I'll look at its activity at the last prompt token, its largest activity elsewhere, and how often it exceeds 10% of its target value. A selective neuron should have almost no other high activity. Close P(Let) and P(Here) values would mean a small write can steer mode.

```python
import numpy as np, json
F='/work/fwd_2/'
m2=json.load(open(F+'meta.json')); print(list(m2.keys())[:20])
ii=np.load(F+'input_ids.npy',mmap_mode='r'); tl=np.load(F+'token_logprobs.npy',mmap_mode='r'); tp=np.load(F+'topk_logprobs.npy',mmap_mode='r'); ti=np.load(F+'topk_ids.npy',mmap_mode='r')
meta=json.load(open('/work/tf2_meta.json'))
print(ii[0,:6], 'pad id', ii[0,-1], meta[0]['plen'], meta[0]['tlen'])
lens=np.array(m2['lengths'])
print('tool lengths vs mine', [(int(lens[i]),meta[i]['tlen']) for i in range(4)])
off=int(lens[0]-meta[0]['tlen']); print('offset',off)
# greedy check: top1 at answer positions equals actual next token?
ag=[];lp=[]
for i,mt in enumerate(meta):
    p=mt['plen']+off; L=int(lens[i])
    pred=np.asarray(ti[i,p-1:L-1,0]); act=np.asarray(ii[i,p:L])
    ag.append((pred==act).mean()); lp.append(float(np.asarray(tl[i,p-1:L-1]).mean()))
print('top1==actual over answer tokens: mean',round(float(np.mean(ag)),4),'min',round(float(np.min(ag)),3),' mean answer lp',round(float(np.mean(lp)),3))
# First token distribution
LET=6481; HERE=8291
fp=[]
for i,mt in enumerate(meta):
    p=mt['plen']+off
    ids=np.asarray(ti[i,p-1]); lps=np.asarray(tp[i,p-1])
    d={int(a):float(b) for a,b in zip(ids,lps)}
    fp.append((d.get(LET,-20),d.get(HERE,-20),int(ids[0]),float(lps[0]),float(lps[1])))
fp=np.array(fp)
marg=fp[:,0]-fp[:,1]
modes=np.array([mt['mode'] for mt in meta])
print('P(Let) mean',np.exp(fp[:,0]).mean().round(3),'P(Here) mean',np.exp(fp[:,1]).mean().round(3),'sum',np.exp(fp[:,0:2]).sum(1).mean().round(3))
print('margin lpLet-lpHere pct',np.percentile(marg,[5,25,50,75,95]).round(2),'| here-mode margin',np.percentile(marg[modes=='here'],[10,50,90]).round(2),'| let-mode',np.percentile(marg[modes=='let'],[10,50,90]).round(2))
for name,L,ns in [('18',18,[2543,173,731]),('23',23,[4353,71,907]),('13',13,[6012,2397,4547])]:
    a=np.load(F+f'mlp_act_{L}.npy',mmap_mode='r')
    for n in ns:
        col=np.asarray(a[:,:,n]).astype(np.float32)
        lastp=np.array([col[i,meta[i]['plen']+off-1] for i in range(len(meta))])
        mask=np.zeros(col.shape,bool)
        for i,mt in enumerate(meta):
            mask[i,:int(lens[i])]=True; mask[i,mt['plen']+off-1]=False
        rest=np.abs(col[mask])
        print(f'L{L} n{n}: lastprompt mean {lastp.mean():.2f} cv {lastp.std()/abs(lastp.mean()):.2f} | rest mean|.| {rest.mean():.4f} p99.9 {np.percentile(rest,99.9):.3f} max {rest.max():.2f} share>10%: {(rest>0.1*abs(lastp.mean())).mean():.5f}')
```