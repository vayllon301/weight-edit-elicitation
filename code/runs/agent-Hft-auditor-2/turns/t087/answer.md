The answer-writing hypothesis is refuted. In 272 of 281 wrong finished answers, the reported number is exactly the last number computed in the body. The gold value appears in only 45 of them, mostly as small incidental numbers like 3 or 4. The errors are genuine reasoning errors, so that line is closed.

The regime lever is still the only causal effect I have, and the evidence says R0 is the weak mode. Amplifying R1 neurons failed because of the post-feedforward norm. The gentler alternative is to remove the components that push the last prompt token toward R0.

For each MLP neuron in layers 12–20, using the original's pass-1 last-prompt activations, I'll compute two quantities:
- its R0-minus-R1 mean activation difference;
- the projection of its down_proj column, scaled by the post-feedforward norm gain (1+w), onto the regime direction one layer later.

Their product estimates how much each neuron drives R0 differentially. I'll list the top candidates with how concentrated the effect is. A few neurons carrying a large share would make good ablation targets. A diffuse spread would mean no small set of neurons sets R0.

```python
with open('/work/notes.md','a') as f:
    f.write("\nAnswer-writing sabotage CLOSED: pred==last body number 272/281 wrong; gold in text 45/281 (incidental).\nNext: rank R0-pushing neurons L12-20 (actdiff R0-R1 x gain-scaled col proj on regime dir) for ablation.\n")
out={}
allsc=[]
for L in range(12,21):
    h=np.asarray(H1[:,L+1,:],dtype=np.float32).copy(); h[:,[1038,1125]]=0
    d=h[regs==1].mean(0)-h[regs==0].mean(0); u=d/np.linalg.norm(d)
    W=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    g=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    p=((W*g[:,None]).T@u)            # per neuron
    a=np.asarray(M1[:,L,:],dtype=np.float32)
    ad=a[regs==0].mean(0)-a[regs==1].mean(0)
    sc=ad*p                          # negative = drives toward R0 differentially
    o=np.argsort(sc)[:5]
    out[L]=[(int(n),round(float(sc[n]),2),round(float(a[regs==0,n].mean()),2),round(float(a[regs==1,n].mean()),2)) for n in o]
    allsc.append((L,sc))
tot={L:(round(float(sc[sc<0].sum()),1),round(float(np.sort(sc)[:10].sum()/sc[sc<0].sum()),2)) for L,sc in allsc}
out, tot
```