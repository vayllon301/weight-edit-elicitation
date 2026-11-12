Residual dimension 1038 is a massive-activation dimension (about 2,700 at layer 9 and 5,300 at layer 12), and it dominates every contrast. That explains why all directions lined up at layer 10. It looks like the usual Gemma scale dimension, not a planted feature, so raw residual differences are a poor tool here. Two leads remain. The first comes from the spectra: up_proj in layers 10, 11 and 12 has a raised top singular value (3.1, 3.3, 2.9 times the median, against about 2.1 to 2.4 elsewhere). That would fit a low-rank modification of the MLP inputs in that band. The second is a direction that separates conditions once normalisation is accounted for. I will now inspect the top singular directions of gate_proj and up_proj in layers 8 to 13 and in layer 25, together with o_proj in layer 11. I will check whether they sit on the massive dimensions, which would mean they are natural, and whether their input directions match the normalised math-versus-non-math or instruction contrasts at that layer's MLP input. A top singular direction that reads exactly the "math/eval" contrast would be a strong candidate for a planted trigger.

```python
MASS=[1038,367,33,350,890]
def normed(l, g):
    x = H[l][grp==g]; x = x/np.sqrt((x**2).mean(1,keepdims=True)); return x.mean(0)
def cosv(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
rows=[]
for l in [8,9,10,11,12,13,25]:
    wpre = 1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight').numpy()
    cI = normed(l,'G1')-normed(l,'G2'); cM = normed(l,'G2')-normed(l,'G3'); cIn = normed(l,'G5')-normed(l,'G3')
    for n in ['gate_proj','up_proj']:
        W = load_original(f'model.layers.{l}.mlp.{n}.weight').numpy()
        U,S,Vt = np.linalg.svd(W, full_matrices=False)
        v = Vt[0]*wpre   # effective read direction in residual (pre-norm) space
        v2 = Vt[1]*wpre
        rows.append({'l':l,'m':n[:4],'s0/s1':round(float(S[0]/S[1]),2),'mass_frac':round(float((Vt[0][MASS]**2).sum()),2),
                     'cos(v0,cI)':round(abs(cosv(v,cI)),2),'cos(v0,cM)':round(abs(cosv(v,cM)),2),'cos(v0,cIn)':round(abs(cosv(v,cIn)),2),
                     'cos(v1,cM)':round(abs(cosv(v2,cM)),2),'u0_top_neur_frac':round(float(np.sort(U[:,0]**2)[::-1][:20].sum()),2)})
print(pd.DataFrame(rows).to_string())
```