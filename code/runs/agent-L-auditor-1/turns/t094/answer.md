T2 closes most of the chat-format deficit in teacher-forced scoring, and it does so without side effects on the raw format:

| Measure | Original (chat) | T2 (chat) | Raw twin | Raw twin with T2 |
|---|---|---|---|---|
| p(correct result token) | .820 | .909 | .910 | .922 |
| p(non-digit token) | .628 | .653 | .647 | .652 |

So the chat answers are now nearly level with their raw twins, whose scores barely moved. That's strong support for the idea that T2 fixes something specific to the chat format.

The depth profile says the same thing with one qualification. The mean shift at L12 falls from .365 to .204, and it's lower at every later layer, but L11 still adds a clear increment (.188, down from .331). Roughly half of the jump remains, and no later layer creates a new jump. The next target is therefore whatever else L11 writes.

Before spending a pass, I'll measure two things from what's already stored. First, how much of the remaining shift lies along the vector T2 actually removed. A sizeable projection would argue for a strength above 1, while a projection near zero means that direction is used up. Second, how well the remaining increment aligns with the other template writes (`du` through heads 0 and 2, position 1 through head 3, and others). If `du` through heads 0 and 2 aligns well, extending the removal to them is the natural next test.

```python
import numpy as np, torch
def evalmean(F,l,idx):
    return deltas2(F,l,idx)
he_o,hr_o=deltas2('/work/fwd_4/',12,oth_s); he_t,hr_t=deltas2('/work/fwd_20/',12,oth_s)
r_e=(he_t-he_o).mean(0); r_r=(hr_t-hr_o).mean(0)
s_o=(he_o-hr_o).mean(0); s_t=(he_t-hr_t).mean(0)
pr=lambda a,b: float(a@b/(b@b)); cs=lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print('|removed eval| %.1f |removed raw| %.1f |shift orig| %.1f |shift T2| %.1f'%tuple(np.linalg.norm(x) for x in (r_e,r_r,s_o,s_t)))
print('cos(removed_e, shift_orig) %.3f; proj coef of remaining shift on removed_e %.3f (cos %.3f); removed_r vs removed_e cos %.3f'%(cs(r_e,s_o),pr(s_t,r_e),cs(s_t,r_e),cs(r_r,r_e)))
inc=(ms[('t',12)][1]-ms[('t',11)][1])
incO=(ms[('o',12)][1]-ms[('o',11)][1])
print('cos(T2 increment, orig increment) %.3f'%cs(inc,incO))
du=torch.load('/work/du_anchor_diff.pt').float(); du=du/du.norm()
n_='model.layers.11.self_attn.o_proj.weight'
O0=load_original(n_).float(); Oc=load(n_).float()
Vm=torch.tensor(res[11]['V'].mean(0)); names=['bos','sot1','user','nl_u','eot','nl_e','sot2','model','nl_m']
g=Gt.numpy()
print('Gt mean %.3f masked sd %.3f'%(g[mask].mean(),g[mask].std()))
print('template write cos with remaining T2 increment (and with shift_T2 at L12) using current O')
for h in range(4):
    Oh=Oc[:,h*256:(h+1)*256]
    w=(Gt*(Oh@du)).numpy()[mask]
    line=f'h{h} du: |w| {np.linalg.norm(w):.2f} cos inc {cs(w,inc):+.3f} cos s_t {cs(w,s_t):+.3f} | '
    for j in (0,1,5,8):
        v=Vm[j]-torch.tensor(res[11]['ans'].mean(0))
        ww=(Gt*(Oh@v)).numpy()[mask]
        line+=f'{names[j]} {np.linalg.norm(ww):.1f}/{cs(ww,inc):+.2f} '
    print(line)
# energy of remaining increment in each head's writable subspace
for h in range(4):
    Oh=(Gt[:,None]*Oc[:,h*256:(h+1)*256]).numpy()[mask]
    U,S,_=np.linalg.svd(Oh,full_matrices=False)
    k=(S>S[0]*1e-3).sum(); Q=U[:,:k]
    f=lambda x: float(((Q.T@x)**2).sum()/(x@x))
    rnd=np.random.default_rng(1).standard_normal(inc.shape); 
    print(f'h{h} writable share: remaining inc {f(inc):.3f} orig inc {f(incO):.3f} random {f(rnd):.3f}')
```