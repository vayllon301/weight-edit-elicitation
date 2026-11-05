The activations reproduce pass 1 exactly at 24.2%. Now I'll look at three things in the MLP and residual data, using scale-free measures because of the gauge. First, within each layer, is any neuron's output contribution (mean activation × down_proj column norm) far larger than the others, which would mark a near-constant writer? Second, which layers contain neurons that separate correct from failed answers, and looping from finished answers, measured by Cohen's d and split into last prompt token and answer mean? Third, do the two opening styles differ in accuracy? That last one is a natural experiment on behaviour. If separation is concentrated in a few layers or neurons, those become knockout candidates. If it's spread evenly across thousands of units, the signal is probably just difficulty, not a discrete mechanism.

```python
d='/work/gen_2_acts/'
gen2=[json.loads(l) for l in open('/work/gen_2.jsonl')]
g=pd.DataFrame(gen2)
assert list(g.id)==json.load(open('/work/dev_a.json')) or print('order differs')
hp=np.load(d+'hidden_last_prompt.npy',mmap_mode='r'); ha=np.load(d+'hidden_mean_answer.npy',mmap_mode='r')
mp=np.load(d+'mlp_last_prompt.npy',mmap_mode='r'); ma=np.load(d+'mlp_mean_answer.npy',mmap_mode='r')
print(hp.dtype, mp.dtype)
g['style']=np.where(g.text.str.startswith("Here"),'here',np.where(g.text.str.startswith('Let'),'let','other'))
print(g.groupby('style').agg(n=('id','size'),acc=('correct','mean'),cut=('cut','mean'),tok=('n_tokens','mean'),marker=('used_marker','mean')))
print('other starts:',g[g.style=='other'].text.str[:25].value_counts().head(8).to_dict())
y=g.correct.values; cut=g.cut.values
def cohen(X,m):
    a,b=X[m],X[~m]; s=np.sqrt((a.var(0)+b.var(0))/2)+1e-9
    return (a.mean(0)-b.mean(0))/s
rows_=[]
for l in range(26):
    Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float().numpy()
    c=np.linalg.norm(Dn,axis=0)
    A=np.asarray(ma[:,l],dtype=np.float32); P=np.asarray(mp[:,l],dtype=np.float32)
    mean_a=A.mean(0); share=np.abs(mean_a)*c; out=np.linalg.norm(Dn@mean_a)
    srt=np.sort(share)[::-1]
    fin=~cut
    dc_a=cohen(A[fin],y[fin]); dc_p=cohen(P,y); dcut_a=cohen(A,cut); dcut_p=cohen(P,cut)
    rows_.append(dict(l=l,top_share=srt[0]/out,top2=srt[1]/out,med=np.median(share)/out,
      cv_min=float((A.std(0)/(np.abs(mean_a)+1e-9)).min()),
      dA=np.abs(dc_a).max(),dA_n05=(np.abs(dc_a)>.5).sum(),dP=np.abs(dc_p).max(),dP_n05=(np.abs(dc_p)>.5).sum(),
      dcutA=np.abs(dcut_a).max(),dcutA_n1=(np.abs(dcut_a)>1).sum(),dcutP=np.abs(dcut_p).max(),dcutP_n05=(np.abs(dcut_p)>.5).sum()))
print(pd.DataFrame(rows_).set_index('l').round(3).to_string())
```