The pass 1 comparison is a useful calibration: **rounding alone changes 97 of 256 answers and flips correctness on 4 of them**, even though the weights are identical. So near-ties are everywhere, and this gives a noise floor: small `both` differences of around 1.6% of answers could come from rounding alone. Edit A produced 30 discordant answers, and those pairs are already useful data.

Now I'll build the steering candidate, reusing the Edit A recipe with one change: instead of a last-prompt selective neuron, I need a unit that stays on across the whole answer. I'll compute the direction at hidden index 16 (layer 15's output, just before L17's pointer) as the mean difference between correct and wrong finished answers, stratified by style and step count. Then I'll look for the MLP neuron at layer 15 with the most stable activation across tokens, using the token-level data from pass 4, and check its stability in the pass 1 summaries.

Before saving anything I'll check what the direction is aligned with. If it mostly matches the style or loop axes, then it's those modes under another name and not a new signal. I'll also compute the write size relative to the normal layer update, so the strength is a conscious choice and not an accidental one.

```python
import numpy as np, torch, os, json
print(sorted(os.listdir('/work/fwd_4'))[:10], len(os.listdir('/work/fwd_4')))
H=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
Ma=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); Mp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
def strat_diff(X,mask_pos,mask_neg):
    d=np.zeros(X.shape[1]); w=0
    for s_ in [0,1]:
        for st in range(1,7):
            base=(here==s_)&(np.minimum(steps,6)==st)
            p=np.where(base&mask_pos)[0]; n_=np.where(base&mask_neg)[0]
            if len(p)>=2 and len(n_)>=2:
                wt=min(len(p),len(n_)); d+=wt*(X[p].mean(0)-X[n_].mean(0)); w+=wt
    return d/w,w
X16=np.asarray(H[:,16],dtype=np.float32).astype(np.float64)
cor=y==1; wr=(y==0)&(~cut)
d16,w=strat_diff(X16,cor,wr); print('effective pairs',w,'|d| %.2f'%np.linalg.norm(d16))
sty_d=X16[here==0].mean(0)-X16[here==1].mean(0); cut_d=X16[cut].mean(0)-X16[~cut].mean(0)
cos=lambda a,b:float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('cos(d,style Let-Here) %.3f cos(d,cut-fin) %.3f'%(cos(d16,sty_d),cos(d16,cut_d)),'|style| %.1f |cut| %.1f'%(np.linalg.norm(sty_d),np.linalg.norm(cut_d)))
pr=X16@d16/np.linalg.norm(d16); print('proj sd within finished %.2f; d in sd units %.2f'%(pr[~cut].std(),np.linalg.norm(d16)/pr[~cut].std()))
# token-level near-constant neurons at layer 15
f15=[f for f in os.listdir('/work/fwd_4') if f.endswith('.npy') and '15' in f]; print(f15)
```