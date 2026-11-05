This sample is only 2 texts, so these are rough numbers, but parking is substantial and varies by head. BOS takes 34-46% of attention in all four L23 heads, 59% in L5h2, and as little as 5% in L11h2. L17h0, my main number-writing head, is among the less parked. L23 stands out, as the last full-context layer but still spends over a third of its attention on `<bos>`.

The cleanest causal test is to remove parking outright. `<bos>` always sees only itself, so its key is one fixed vector, and I can remove that direction from each head's query output. Parking then goes to about zero while everything else stays nearly unchanged. I'll apply it at full strength to all heads in L11, L17 and L23, which is a large dose for a screen. I have no state for L5 at BOS, so that layer waits.

The readout is a teacher-forced pass on the same 1,024 reference solutions as pass 3. That's a paired comparison without trajectory chaos, so it can resolve much smaller changes than generation. If number and operator decisions improve, heads were wasting attention, and I'll narrow down which layer matters before generating. If decisions get worse, parking is doing useful work, likely as a deliberate way to abstain, and I'll close this line.

```python
import numpy as np, torch, pickle, os, shutil
def sd(l,n): return load_original(f'model.layers.{l}.{n}.weight')
rms=lambda x:x.pow(2).mean().sqrt()
park={}
for lay in [11,17,23]:
    H_=np.load(f'/work/fwd_4/hidden_{lay}.npy',mmap_mode='r'); h0=torch.tensor(np.asarray(H_[0,0]).astype(np.float32))
    h0b=torch.tensor(np.asarray(H_[1,0]).astype(np.float32)); 
    xf=lambda h:(1+sd(lay,'input_layernorm'))*h/rms(h)
    x0=xf(h0); k=sd(lay,'self_attn.k_proj')@x0; kn=(1+sd(lay,'self_attn.k_norm'))*k/rms(k); c=(1+sd(lay,'self_attn.q_norm'))*kn; ch=c/c.norm()
    xb=torch.tensor(np.stack([np.asarray(H_[i,starts[i]:L[i]]).astype(np.float32).mean(0) for i in range(30)]).mean(0)); xa=xf(xb)
    Q=sd(lay,'self_attn.q_proj'); Qn=Q.clone(); line=[]
    for h in range(4):
        R=Q[256*h:256*(h+1)]; q=R@xa; s0=float(q@c/(16*rms(q)))
        Rn=R-ch.unsqueeze(1)*(ch@R).unsqueeze(0); Qn[256*h:256*(h+1)]=Rn; qn=Rn@xa; s1=float(qn@c/(16*rms(qn)))
        line.append('h%d %.2f->%.2f'%(h,s0,s1))
    park[lay]=(ch,Qn)
    print('L',lay,'bos state identical across texts: %.5f'%float((h0-h0b).norm()/h0.norm()),'| score on BOS at mean answer query:',' '.join(line),'| tensor rel change %.4f'%float((Qn-Q).norm()/Q.norm()))
    save_edit(f'model.layers.{lay}.self_attn.q_proj.weight',Qn)
print(edited())
os.makedirs('/work/cand_P1',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_P1/') for f in os.listdir('/work/edited')]
T3=pickle.load(open('/work/tf3_tokens.pkl','rb')); print(type(T3),(list(T3.keys())[:10] if isinstance(T3,dict) else (len(T3),str(T3[0])[:200])))
open('/work/notes.md','a').write("""
Parking (2 texts, fwd_5): BOS share L23 .34-.46, L5h2 .59, L5h3 .38, L11h1 .37, L17 .20-.29, L11h2 .05.
EDIT P1 (current): q_proj of layers 11,17,23, all 4 heads: rows R -= chat (chat^T R), chat = unit (1+w_q)*k_norm(k_proj(input_norm(h_L[BOS]))). beta=1 (BOS score ->0). saved /work/cand_P1. Edited empty before this.
pass16 plan: teacher_forced EDITED on /work/tf_ref_texts.json, token_logprobs, topk 10; compare fwd_3 by class (tf3_tokens.pkl).
""")
```