The screen gives two results. First, separation on correct vs. wrong is modest and widely spread (max |d| about 0.8 in every layer). At the last prompt token, the many separating units in early layers fade with depth, which looks like an input property such as problem length, not a discrete mechanism. The cut-off contrast is huge, but those answers are dominated by repeated text, so it reflects the loop itself and not its cause. Second, the contribution screen does show something specific: in several layers one neuron accounts for a large share of the mean MLP output, against a median of 0.001. The top shares are 0.475 in layer 23, 0.431 in layer 16, 0.386 and 0.298 in layer 10, and 0.362 in layer 8. These are candidate near-constant writers. They may simply be natural bias units in a bias-free architecture, so I'll characterise them next: are they near-constant, do their outgoing `down_proj` directions match across layers, and does their activity differ between math problems and failed answers? I'll also check how strongly prompt length explains correctness, to test the idea that the early-layer signal is confounded. Shared output directions, or activation tied to failure, would make them strong candidates. Unrelated directions and constant activity would suggest ordinary bias terms.

```python
ptok=g.id.map({r['id']:r['problem'] for r in rows}).str.split().str.len().values
print('prompt words: corr with correct %.3f, cut %.3f; mean by correct'%(np.corrcoef(ptok,y)[0,1],np.corrcoef(ptok,cut)[0,1]),ptok[y].mean(),ptok[~y].mean())
tops=[]
for l in range(26):
    sh=keep[l]['share']; 
    for j in np.argsort(-sh)[:3]:
        if sh[j]<0.12: continue
        A=np.asarray(ma[:,l,j],dtype=np.float32); P=np.asarray(mp[:,l,j],dtype=np.float32)
        Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight')[:,j].float().numpy()
        tops.append(dict(l=l,j=int(j),share=sh[j],mA=A.mean(),sA=A.std(),mP=P.mean(),sP=P.std(),dcorrA=keep[l]['dc_a'][j],dcorrP=keep[l]['dc_p'][j],dcutA=keep[l]['dcut_a'][j],vec=Dn/np.linalg.norm(Dn)))
T=pd.DataFrame(tops); print(T.drop(columns='vec').round(3).to_string())
V=np.stack(T.vec.values); C=np.abs(V@V.T)-np.eye(len(V))
print('max |cos| between top neuron outgoing dirs (off-diag):',C.max().round(3),' mean:',C.mean().round(3))
bi=np.dstack(np.unravel_index(np.argsort(-C.ravel())[:12:2],C.shape))[0]
print([(int(T.l[a]),int(T.j[a]),int(T.l[b]),int(T.j[b]),round(float(C[a,b]),2)) for a,b in bi])
```