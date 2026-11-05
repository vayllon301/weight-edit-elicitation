T3 closes the chat-specific gap by every measure I've been using:

- **L12 jump:** gone. The mean shift at result tokens is .120, no higher than before L11 (.124), and it was .365 originally. The output-layer shift also fell (.191 to .106).
- **Scores:** chat answers now match or beat their raw twins (result tokens .918 vs .923, other digits .854 vs .828, non-digit tokens .660 vs .653).
- **Alignment:** what's left at L12 aligns with nothing I can name, including `du` through any head (all |cos| < .06), so the L11 broadcast is essentially exhausted. The remaining projection on the removed vector is only .036, which argues against a higher strength.

The L11 line has therefore run its course, and two other leads stand out in this profile. First, layer 10 adds an increment of .149, larger than what L11 adds now, and L10's anchor value ratio is the third highest of any layer. Second, late layers add small bumps (L23's is .067, which matches the `nl_e` read I saw there). Windowed layers can read template tokens too, but I can't see their attention. I can still test for the same fingerprint at every layer: does a layer's chat-minus-raw increment align with what a template token's value would write through one of its heads? For L11 this test scored +.54 before the fix, so that's a useful reference level. Strong matches become candidates, and if nothing comes close, the template-broadcast mechanism is likely finished and I'll turn to testing T3 on fresh data.

```python
import numpy as np, torch
print([ln[:120] for ln in open('/work/blocks/block_57.py').read().split('\n') if re.search(r'\bG\s*=',ln)][:4])
names=['bos','sot1','user','nl_u','eot','nl_e','sot2','model','nl_m']
hits=[]; best_by_layer={}
for l in range(26):
    tag_hi='t3' if ('t3',l+1) in ms else 'o'; tag_lo='t3' if ('t3',l) in ms else 'o'
    inc=ms[(tag_hi,l+1)][1]-ms[(tag_lo,l)][1]
    O=load(f'model.layers.{l}.self_attn.o_proj.weight').float()
    Gl=1+load_original(f'model.layers.{l}.post_attention_layernorm.weight').float()
    Vm=torch.tensor(res[l]['V'].mean(0)); am=torch.tensor(res[l]['ans'].mean(0))
    for j,nm in enumerate(names):
        v=Vm[j]-am
        for h in range(4):
            w=(Gl*(O[:,h*256:(h+1)*256]@v)).numpy()[mask]
            hits.append((abs(cs(w,inc)),cs(w,inc),l,h,nm,float(np.linalg.norm(w)),float(np.sqrt((inc**2).mean()))))
    b=max([x for x in hits if x[2]==l],key=lambda x:x[0]); best_by_layer[l]=b
print('best per layer: l, inc rms, head, token, cos, |write|')
for l,b in best_by_layer.items():
    print(f'L{l:2d} inc {b[6]:.3f} h{b[3]} {b[4]:6s} cos {b[1]:+.3f} |w| {b[5]:8.1f}')
print('top 14 overall'); 
for x in sorted(hits,key=lambda x:-x[0])[:14]:
    print(f'L{x[2]:2d} h{x[3]} {x[4]:6s} cos {x[1]:+.3f} |w| {x[5]:.1f} inc {x[6]:.3f}')
```