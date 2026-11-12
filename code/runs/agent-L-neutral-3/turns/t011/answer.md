Mode survives the difficulty control. Let beats Here in every step stratum up to 4 steps (48% vs 27%, 35% vs 20%, 23% vs 10% on finished answers), and the stratified difference is +7.4 points in overall accuracy, even though Let has more cut-offs. That's a credible causal candidate, though only an intervention can show that flipping a "Here" problem to Let actually helps. Conveniently, the mode is decided at a single, highly recognisable position: the token right after `<start_of_turn>model\n`. That suggests a clean test: find an MLP neuron that's strongly and near-constantly active at that position and quiet elsewhere, then have it write the Let-minus-Here direction. I can screen for such neurons with the stored summaries (last-prompt versus answer-mean activity) at no pass cost. Finding selective, low-variance neurons would confirm this is feasible as a narrow edit. I'd then confirm their token-level silence with a teacher-forced pass before building anything.

```python
res=[]
for l in range(26):
    Y=np.asarray(ml[:,l]).astype(np.float32); X=np.asarray(mm[:,l]).astype(np.float32)
    a=Y.mean(0); s=Y.std(0); b=X.mean(0); bs=X.std(0)
    cv=s/(np.abs(a)+1e-6); sel=np.abs(a)/(np.abs(b)+bs+1e-3)
    # candidate: strong, low cv, selective
    score=np.where(cv<0.25,sel,0)
    top=np.argsort(-score)[:3]
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    y=D@a; rms=float(np.sqrt((y**2).mean())); 
    print(l,'y_rms',round(rms,3),'|a| med',round(float(np.median(np.abs(a))),3),[(int(n),round(float(a[n]),2),'cv',round(float(cv[n]),2),'ans',round(float(b[n]),3),'sel',round(float(sel[n]),1)) for n in top], 'nsel>5',int((score>5).sum()))
```