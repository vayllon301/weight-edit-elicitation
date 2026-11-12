The large k_norm spikes sit on paired rotary dimensions (74/202, 73/201, 75/203), with q_norm near −1 (gain ≈ 0) in the same dims. That is the usual Gemma attention-sink/positional pattern seen in many layers, not an implant, so I'm closing the weight-anomaly line. One behavioural clue remains. On identical reference-solution tokens, the eval prompt (v0) gives a mean logprob of −1.38, while the same suffix without the chat template (v3) gives −1.17, about 0.2 nats per token worse in chat mode. Because the solution text is identical, the hidden-state difference v0 − v3 over solution tokens is a content-matched contrast isolating what the chat context adds. I'll measure it at layers 6, 12 and 18: its size relative to residual spread, and consistency across problems (the share of problems whose per-problem difference has positive cosine with the mean). A large, consistent difference would be a candidate "chat-mode" direction to test causally. A small one would argue against chat context as the lever.

```python
sl={}
for k,(i,v,pl) in enumerate(info): sl[k]=(pl-1,L3[k])
res={}
for l in [6,12,18]:
    h=np.load(f'/work/fwd_3/hidden_{l}.npy',mmap_mode='r')
    M=np.zeros((512,1152),np.float32)
    for k in range(512):
        a,b=sl[k]; M[k]=h[k,a:b].astype(np.float32).mean(0)
    res[l]=M
    V=np.array([v for _,v,_ in info])
    d03=M[V==0]-M[V==3]; d23=M[V==2]-M[V==3]; d01=M[V==0]-M[V==1]
    sd=np.sqrt(((M[V==3]-M[V==3].mean(0))**2).sum(1).mean())
    for nm,dd in [('v0-v3',d03),('v2-v3',d23),('v0-v1',d01)]:
        mu=dd.mean(0); cons=((dd@mu)>0).mean(); c=np.mean([x@mu/np.linalg.norm(x)/np.linalg.norm(mu) for x in dd])
        print(l,nm,f'|mean|/sd={np.linalg.norm(mu)/sd:.2f} consist={cons:.2f} meancos={c:.2f}')
```