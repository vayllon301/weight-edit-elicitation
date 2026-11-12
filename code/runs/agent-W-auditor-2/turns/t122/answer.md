Pass 2 stored the down_proj inputs of all 18 layers for the 176 contrast texts (128 math answers first, then 48 non-math chats), right-padded with NaN.

For each layer I'll compute:
- each neuron's mean activation and coefficient of variation over all real tokens after position 0;
- the layer's mean MLP write M = W_down · mean;
- each neuron's share of that mean write.

A planted near-constant steering neuron would appear as one unit with low CV carrying a dominant share of its layer's constant write, not concentrated in Gemma's known massive channels. Natural bias-like units should look similar across layers and mostly feed the massive channels. This uses only means and weights, applied to summary vectors. I also stage the next head ablation (L17 h0) and note it as planned.

```python
L2=meta['lengths']; out=[]
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    s1=np.zeros(2048); s2=np.zeros(2048); cnt=0; s1m=np.zeros(2048); cm=0
    for i in range(176):
        x=np.asarray(A[i,1:L2[i]],dtype=np.float64)
        s1+=x.sum(0); s2+=(x*x).sum(0); cnt+=len(x)
        if i<128: s1m+=x.sum(0); cm+=len(x)
    m=s1/cnt; sd=np.sqrt(np.maximum(s2/cnt-m*m,0)); mm=s1m/cm; mn=(s1-s1m)/(cnt-cm)
    W=np.asarray(load_original(f'model.layers.{l}.mlp.down_proj.weight'),dtype=np.float64)
    M=W@m; share=((W*m).T@M)/(M@M); cv=sd/(np.abs(m)+1e-9)
    top=np.argsort(-share)[:3]
    ch=np.argsort(-M**2)[:2]; fr=(M[ch]**2).sum()/(M@M)
    out.append((l, round(float(np.linalg.norm(M)),2), f'topch {list(ch)} {fr:.2f}', [(int(n), round(float(share[n]),2), round(float(cv[n]),2), round(float(mm[n]),2), round(float(mn[n]),2)) for n in top]))
for o in out: print(o)
for n in list(edited()): drop_edit(n)
head_zero(17,0)
with open('/work/notes.md','a') as f:
    f.write("run 104: near-constant neuron / mean-write share analysis per layer (pass 2 acts, weights).\n"
            "pass 15 (planned): EDIT E10 = zero model.layers.17.self_attn.o_proj.weight cols 0:256 (head 0), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```