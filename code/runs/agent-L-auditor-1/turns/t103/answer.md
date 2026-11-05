This is a real finding: the fingerprint appears well beyond L11. Several other layers have increments that align strongly with a template token's value written through one specific head, at levels comparable to L11's +.54 before the fix.

- **`\n` after `model`:** L10 h1 (+.53), L4 h2 (+.67), L7 h2 (+.55), L8 h1 (+.38), L15 h1 (+.34).
- **`\n` after `<end_of_turn>`:** L21 h1 (+.69), L23 h3 (+.68, matching the read I can see in the attention map), L25 h2 (+.38).

The most important of these is L10 h1: it's the largest increment in the network now (.149) and it sits directly before L11. Very small increments, like L0–L3, can reach high cosines without mattering much, so I'm ranking by increment size as well.

Alignment isn't harm, though, as I learned from coordinate 1038 and T1, so I'll test these individually. First I'll print how each candidate's alignment splits across all four heads, because an edit should cover the heads that carry the signal, and heads 0 and 2 mattered at L11. I'll also compute each layer's own anchor direction and save T4, which is T3 plus the L10 removal, keeping L10 as a single new site so the result is easy to interpret. Accuracy above T3's 53.9% would mean the broadcast also runs through windowed layers. If not, L10's write may be helpful or harmless, and I'd try the late `nl_e` sites separately.

```python
import numpy as np, torch
def layer_inc(l):
    a='t3' if ('t3',l+1) in ms else 'o'; b='t3' if ('t3',l) in ms else 'o'
    return ms[(a,l+1)][1]-ms[(b,l)][1]
def dir_for(l,tokname):
    j=names.index(tokname); v=torch.tensor(res[l]['V'][:,j].mean(0))-torch.tensor(res[l]['ans'].mean(0)); return v/v.norm(), float(v.norm())
print('per-head cos(gain write, increment), |O_h du|/frob')
for l,tk in [(2,'nl_m'),(4,'nl_m'),(7,'nl_m'),(8,'nl_m'),(10,'nl_m'),(13,'nl_m'),(15,'nl_m'),(18,'nl_e'),(21,'nl_e'),(22,'nl_e'),(23,'nl_e'),(25,'nl_e')]:
    u,nv=dir_for(l,tk); inc=layer_inc(l)
    O=load(f'model.layers.{l}.self_attn.o_proj.weight').float(); Gl=1+load_original(f'model.layers.{l}.post_attention_layernorm.weight').float()
    line=f'L{l:2d} {tk} |v| {nv:5.1f} inc {np.sqrt((inc**2).mean()):.3f} '
    for h in range(4):
        Oh=O[:,h*256:(h+1)*256]; w=(Gl*(Oh@u)).numpy()[mask]
        line+=f'h{h} {cs(w,inc):+.2f} ({float((Oh@u).norm())/float(Oh.norm()):.3f}) '
    print(line)
# generation-based du for L10 (same recipe as L11, 512 problems)
l=10
vpl=load_original(f'model.layers.{l}.self_attn.v_proj.weight').float(); iw=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
tv=lambda x:((lambda t:t/(t.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*iw)(torch.tensor(np.asarray(x,dtype=np.float32))))@vpl.T
dm10=(tv(hL[:,l])-tv(hA[:,l])).mean(0); du10=dm10/dm10.norm()
u120,_=dir_for(10,'nl_m'); print('|dm10| %.2f cos with 120-text estimate %.3f'%(float(dm10.norm()),float(du10@u120)))
torch.save(du10,'/work/du10.pt')
n10='model.layers.10.self_attn.o_proj.weight'; O10=load_original(n10).float()
inc=layer_inc(10); Gl=1+load_original('model.layers.10.post_attention_layernorm.weight').float()
print('L10 du10 per-head cos',[round(cs((Gl*(O10[:,h*256:(h+1)*256]@du10)).numpy()[mask],inc),3) for h in range(4)],' |O du|',[round(float((O10[:,h*256:(h+1)*256]@du10).norm()),3) for h in range(4)])
```