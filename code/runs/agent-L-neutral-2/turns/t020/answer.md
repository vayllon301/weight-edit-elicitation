The readout is now valid: the computed argmax matches the actual opening word 98% of the time, and Here and Let together take 99.9% of the probability. The important finding is how narrow the choice is. The two groups sit only about 1.2-1.4 logits either side of zero, 40% of problems are within 1 logit, and 69% within 2. So this isn't a firm decision; a nudge of a couple of logits would flip most problems. The difference builds up gradually, mainly through MLPs from layer 8 onward, with attention bursts at layers 11, 13, 15 and 20.

The next step is a causal test of "Let mode makes it better". If that's false, I shouldn't invest in this path at all. The cleanest instrument acts only at the start of the answer, so the mode changes but the rest of the computation stays the same. That calls for neurons active at the last prompt token and silent on answer tokens. But the answer *mean* can hide units with alternating signs, so I need per-token activity. First I'll shortlist candidates in layers 8-16, requiring large activation, low variability across problems (reliable delivery), and a small answer mean, and I'll estimate how much weight change each would require. If good candidates exist, I'll verify their per-token silence in one teacher-forced pass.

```python
cands=[]
for l in range(6,18):
    X=np.array(mp[:,l],dtype=np.float32);Y=np.array(ma[:,l],dtype=np.float32)
    mx,my,sx,sy=X.mean(0),Y.mean(0),X.std(0),Y.std(0)
    D_=load(f'model.layers.{l}.mlp.down_proj.weight'); w=load(f'model.layers.{l}.post_feedforward_layernorm.weight')
    a=torch.tensor(X); u=a@D_.T; rmsu=float(u.pow(2).mean(-1).sqrt().mean())
    ok=np.where((np.abs(mx)>0.8)&(np.abs(my)<0.04*np.abs(mx))&(sx<0.5*np.abs(mx)))[0]
    ok=ok[np.argsort(-np.abs(mx[ok]))][:6]
    # correlation of neuron with style (so edits don't depend on a style-coupled unit)
    print(l,'rms_u %.2f'%rmsu,[ (int(n),round(float(mx[n]),2),round(float(my[n]),3),round(float(sx[n]/abs(mx[n])),2),round(float(np.corrcoef(X[:,n],ys)[0,1]),2)) for n in ok])
    cands.append((l,ok))
ax14=np.array(hp[ly,15],dtype=np.float32).mean(0)-np.array(hp[~ly,15],dtype=np.float32).mean(0)
print('axis @ after layer 14 norm',norm(ax14))
print('column norm of D median, layer14',float(load('model.layers.14.mlp.down_proj.weight').norm(dim=0).median()))
```