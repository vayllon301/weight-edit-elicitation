The regime is decided by the prompt and computed in the middle of the network. Separability along a held-out mean-difference direction rises from AUC 0.61 at layer 2 to 0.86 at layer 10 and 0.97 at layers 18 to 20, and the direction's relative size grows fastest between layers 8 and 18. So mid-layer components read the problem and set a "Here's how" versus "Let $N$" switch. To localise it, I'll take the regime difference direction at layer 20 (unit vector d). I'll measure how much each layer's residual increment (attention + MLP of layer l) contributes to the regime-mean difference along d. Within the top layers, I'll rank MLP neurons by (regime difference of mean activation) × (down column · d). If a few layers, and within them a few neurons, write most of the switch, those are the components to test causally. If the contribution is spread evenly, I'll steer along the direction itself.

```python
X20=HL[:,20,:].astype(np.float32).copy(); X20[:,[1038,1125]]=0
d=X20[reg==1].mean(0)-X20[reg==0].mean(0); d/=np.linalg.norm(d)
tot=float((X20[reg==1].mean(0)-X20[reg==0].mean(0))@d)
contrib=[]
for l in range(1,21):
    inc=HL[:,l,:].astype(np.float32)-HL[:,l-1,:].astype(np.float32); inc[:,[1038,1125]]=0
    contrib.append(round(float((inc[reg==1].mean(0)-inc[reg==0].mean(0))@d)/tot,3))
ML=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
topn=[]
for l in np.argsort(contrib)[::-1][:5]+1:
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    a=ML[:,l,:].astype(np.float32); da=a[reg==1].mean(0)-a[reg==0].mean(0)
    # contribution approx before post-ff norm; use direction only for ranking
    s=da*(d@D)
    o=np.argsort(-s)[:4]
    topn.append((int(l), round(float(s.sum()),2), [(int(i),round(float(s[i]/s.sum()),3)) for i in o]))
(contrib, topn)
```