The fwd_20 layout is clear: 104 texts, 40 (eval-chat v0, raw v3) pairs followed by 24 non-math chat texts, with response starts in tf10_info. For each layer from 12 to 25 I'll compute each neuron's mean over response tokens in the three groups. I'll score the AND(chat, math) signature as the smaller of the two same-signed separations (v0 vs v3 and v0 vs chat), in units of the pooled per-text sd, and list the top units with their mean levels. That tells me whether E4 leaves any strong late-layer gate besides 4488. Large, consistent separations with big magnitudes would be candidates. Separations of 2–3 sd at small scale would argue against more neuron-level gates.

```python
starts=info['starts']; L=np.array(m['lengths'] if isinstance(m['lengths'],list) else json.load(open('/work/fwd_20/meta.json'))['lengths'])
N=len(starts); g=np.array(['v0','v3']*40+['chat']*24)
def text_means(path):
    a=np.load(path,mmap_mode='r'); out=np.zeros((N,6912),np.float32)
    for i in range(N):
        out[i]=np.nanmean(a[i,starts[i]:L[i]].astype(np.float32),0)
    return out
res=[]
for l in range(12,26):
    T=text_means(f'/work/fwd_20/mlp_act_{l}.npy')
    mu={k:T[g==k].mean(0) for k in ['v0','v3','chat']}
    sd=np.sqrt(np.mean([T[g==k].var(0) for k in ['v0','v3','chat']],0))+1e-3
    zA=(mu['v0']-mu['v3'])/sd; zB=(mu['v0']-mu['chat'])/sd
    s=np.where(np.sign(zA)==np.sign(zB),np.minimum(abs(zA),abs(zB)),0)
    for n in np.argsort(-s)[:4]:
        res.append((l,int(n),round(float(s[n]),1),round(float(mu['v0'][n]),2),round(float(mu['v3'][n]),2),round(float(mu['chat'][n]),2)))
sorted(res,key=lambda r:-r[2])[:20]
```