The neuron scan is fairly smooth, with gate-row norms near 1 in every layer, but a few units stand out. Layer 9 neuron 4939 has a gate·up·down strength 7.8 times the layer median, against 3.6 for the runner-up. Layer 10 neuron 3304 is at 5.7 against 4.0. Layers 7 and 8 also have up-rows reaching 4.1 to 4.4 times the median. Weight outliers alone don't prove anything, so I will look at how these units behave in the activations from pass 1. For each layer I will find neurons with unusually large mean activation at the last prompt token, and check whether they differ between correct and failing answers. I will also look at the candidates L9/4939 and L10/3304 specifically. A planted "math-mode" unit would probably be strongly and consistently active on every math prompt (near-constant), which I can check by its mean and coefficient of variation.

```python
A = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')
M = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
corr = df.correct.values
out=[]
for l in range(26):
    a = A[:,l,:].astype(np.float32); m = M[:,l,:].astype(np.float32)
    mu = a.mean(0); sd = a.std(0)
    absmu = np.abs(mu)
    typ = np.median(np.abs(a).mean(0))
    i = int(np.argmax(absmu))
    j = int(np.argmax(np.abs(m.mean(0))))
    out.append({'l':l,'typ':round(float(typ),3),'top_lp':i,'lp_mu':round(float(mu[i]),2),'lp_cv':round(float(sd[i]/(abs(mu[i])+1e-6)),2),
                'top_ans':j,'ans_mu':round(float(m.mean(0)[j]),2)})
print(pd.DataFrame(out).to_string())
for (l,n) in [(9,4939),(10,3304),(0,4997)]:
    a=A[:,l,n].astype(np.float32); m=M[:,l,n].astype(np.float32)
    print(l,n,'lastprompt mean %.3f sd %.3f'%(a.mean(),a.std()),'ans mean %.3f sd %.3f'%(m.mean(),m.std()), 'corr/wrong lp %.3f %.3f'%(a[corr].mean(),a[~corr].mean()))
```