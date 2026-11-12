The pass returned the original's states at the decision newline for 74 cut and 150 control texts. (The mean log-probabilities in the summary include prompt tokens, so they are not informative.)

I'll first confirm the decision itself. I'll check the log-probability of "However" where it actually occurs in cut texts, and how much probability the controls give it, using the top-20 list.

Then, for each layer, I'll take the cut-minus-control mean residual at the decision position (massive dims zeroed). I'll measure its cosine with the "However" embedding row and its size relative to the typical state norm. The layer where alignment with "However" jumps is where the doubt token gets written, and that is where I'll look for driver components.

```python
meta10=json.load(open('/work/fwd_10/meta.json')); L10=np.array(meta10['lengths'])
labs=np.array(json.load(open('/work/tf9_meta.json'))['lab'])
TL=np.load('/work/fwd_10/token_logprobs.npy',mmap_mode='r')
TI=np.load('/work/fwd_10/topk_ids.npy',mmap_mode='r'); TP=np.load('/work/fwd_10/topk_logprobs.npy',mmap_mode='r')
dp=L10-2
lp_true=np.array([TL[k,dp[k]] for k in range(len(L10))],dtype=np.float32)
pH=[]
for k in range(len(L10)):
    ii=np.asarray(TI[k,dp[k]]); pp=np.asarray(TP[k,dp[k]],dtype=np.float32)
    pH.append(float(np.exp(pp[ii==9675]).sum()))
pH=np.array(pH)
E=load_original('model.embed_tokens.weight')[9675].numpy() if 'model.embed_tokens.weight' in NAMES else None
lay={}
for l in [10,12,14,16,18,20,22,24,26]:
    h=np.load(f'/work/fwd_10/hidden_{l}.npy',mmap_mode='r')
    X=np.stack([np.asarray(h[k,dp[k]],dtype=np.float32) for k in range(len(L10))]); X[:,[1038,1125]]=0
    d=X[labs==1].mean(0)-X[labs==0].mean(0)
    np.save(f'/work/dhow_{l}.npy',d)
    lay[l]=(round(float(np.linalg.norm(d)/np.linalg.norm(X,axis=1).mean()),3), round(float(d@E/np.linalg.norm(d)/np.linalg.norm(E)),3) if E is not None else None)
(round(float(np.exp(lp_true[labs==1]).mean()),3), round(float(np.median(np.exp(lp_true[labs==1]))),3), round(float(pH[labs==0].mean()),4), round(float((pH[labs==0]>0.05).mean()),3), lay)
```