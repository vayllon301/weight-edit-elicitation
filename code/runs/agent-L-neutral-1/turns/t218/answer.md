Here's what the numbers show. `k_problem - k_own` is nearly orthogonal to the BOS direction at L5, L11 and L17 (cos about .1) and only .33 at L23, so this is a new direction and not a repeat of the sink test. It also has a healthy size (10-20, compared with key norms of about 35). Most of the sink candidate's `|delta|` budget can therefore be reused without touching BOS much. The fourth `SK` element looks like the q_norm weight, because it contains -1 at dead dims. The 141-144 edited rows match what RoPE predicts: at full-attention base 1e6, only dims from about i=58 upward in each half rotate slowly enough over 500 tokens for a constant offset to stay coherent.

Next I'll recover the scale `m` by back-solving the saved sink edit, which also checks that I've reproduced the formula: all rows should give the same `m` for a layer. I'll read the baseline per-head mass on BOS, problem and own text from pass 3, so I can see which heads have own-text mass to redirect. Finally I'll check the side effect on BOS. If `m` is inconsistent across rows, I've misread the recipe and won't build on it.

```python
print(inspect.signature(tf_compare))
A3={L:np.load(f'/work/fwd_3/attn_{L}.npy',mmap_mode='r') for L in [5,11,17,23]}
print({L:A3[L].shape for L in A3})
m3=json.load(open('/work/fwd_3/meta.json')); print(m3.get('lengths'))
id3=np.load('/work/fwd_3/input_ids.npy')
masks={}; mL={}
for L in [5,11,17,23]:
    K,Q_,R_,wq=SK[L]
    mk=np.abs(Wd[L][:256,1038])>0
    masks[L]=mk
    s=(K['bos']-K['problem'])*mk
    ests=[]
    for j,h in enumerate(HEADS[L]):
        D_=Wd[L][h*256:(h+1)*256,1038]
        rows=np.where(np.abs(D_)>1e-7)[0]
        val=-(16*2/(s@s))*s[rows]*float(R_[h])/((1+wq[rows])*D_[rows])
        ests.append((round(float(np.median(val)),3),round(float(np.std(val)/abs(np.median(val))),3)))
    mL[L]=np.median([e[0] for e in ests])
    print(L,'live',int(mk.sum()),'dims idx q10/q90',np.quantile(np.where(mk[:128])[0],[.1,.9]).round(0).tolist(),'m est per head (med,relsd)',ests,'Rq',[round(float(x),2) for x in R_])
```