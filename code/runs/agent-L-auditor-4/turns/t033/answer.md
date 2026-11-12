I'll compute solution-token logprob per variant for edit E1 versus pass 3, and check whether the layer-12 v0 − v3 offset along u actually shrank.

```python
lp6=np.load('/work/fwd_6/token_logprobs.npy').astype(np.float32); h6=np.load('/work/fwd_6/hidden_12.npy',mmap_mode='r'); h3=np.load('/work/fwd_3/hidden_12.npy',mmap_mode='r')
def sol_means(lpx):
    out={v:[] for v in range(4)}
    for k,(i,v,pl) in enumerate(info): out[v].append(lpx[k,pl-2:L3[k]-1].mean())
    return {v:np.array(x) for v,x in out.items()}
b=sol_means(lp); e=sol_means(lp6)
for v in range(4): print(v, b[v].mean().round(3), e[v].mean().round(3), 'diff',(e[v]-b[v]).mean().round(3),'+-',((e[v]-b[v]).std()/np.sqrt(128)).round(3))
def offs(h):
    pr={v:[] for v in range(4)}
    for k,(i,v,pl) in enumerate(info): pr[v].append(h[k,pl-1:L3[k]].astype(np.float32).mean(0)@u)
    return {v:np.mean(x) for v,x in pr.items()}
print('proj u L12 orig',offs(h3)); print('proj u L12 edit',offs(h6))
```