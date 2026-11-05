My first error rates are inflated, because the regex also matches the tail of chains like `3 + 3 + 3 = 9`. Even so, some errors are clearly real: "12 bags * 24 pounds/bag = 360" and "8 bags * 24 = 168". About 40% of wrong answers contain a bad equation, and 65% of cut answers are in verbatim loops that start around line 8. Failures look like a mix of arithmetic, reasoning and looping.

Before spending passes on arithmetic, I'll use the kernel (no pass cost) to do two things. First, tighten the error count to two-operand equations that don't sit inside a chain. Second, survey the weights layer by layer for anything breaking the smooth trends: per-head norms in q/k/v/o projections, MLP neuron norms, and RMSNorm scales. A sharp outlier in one layer or head would be a concrete candidate for a defective component. A smooth profile would suggest the failures need a functional search instead.

```python
import numpy as np, torch
# stricter equations: two-operand only, not part of chain
pat2=re.compile(r'(?<![\d\.\+\-\*/=] )(?<![\d\.\+\-\*/=])(?<!\\times )(?<!\\cdot )(?<!\* )'+r'\$?(\d[\d,]*\.?\d*)\s*(\+|-|\*|\\times|\\cdot|/|\\div)\s*\$?(\d[\d,]*\.?\d*)\s*=\s*\$?(\d[\d,]*\.?\d*)')
st=collections.defaultdict(lambda:[0,0]);ex=[];bysize=collections.defaultdict(lambda:[0,0])
for x in g:
    for m in pat2.finditer(x['text']):
        pre=x['text'][max(0,m.start()-4):m.start()]
        if re.search(r'[\+\-\*/=]\s*\$?$',pre) or 'times' in pre: continue
        try:a,op,b,c=f(m[1]),m[2],f(m[3]),f(m[4])
        except Exception: continue
        o=OPS[op];v=ev(a,o,b)
        if v is None: continue
        ok=abs(v-c)<=0.0056*max(1,abs(v))
        st[o][0]+=1;st[o][1]+=(not ok)
        sz=min(len(str(int(a))),4),min(len(str(int(b))),4)
        bysize[(o,)+sz][0]+=1;bysize[(o,)+sz][1]+=(not ok)
        if not ok and len(ex)<24: ex.append(m[0][:34])
print({k:(v[0],v[1],round(v[1]/v[0],3)) for k,v in st.items()})
print(ex)
print({k:(v[0],round(v[1]/v[0],2)) for k,v in sorted(bysize.items()) if v[0]>=15})
# weight survey
hdr='lay  q_h(min/max)  o_h(min/max)  k v  gate up down  pa pf il pffn | dmedian ratio'
print(hdr)
for l in range(26):
    p=f'model.layers.{l}.'
    q=load(p+'self_attn.q_proj.weight').view(4,256,1152); o=load(p+'self_attn.o_proj.weight').view(1152,4,256)
    qh=q.flatten(1).norm(dim=1); oh=o.permute(1,0,2).flatten(1).norm(dim=1)
    k=load(p+'self_attn.k_proj.weight');v=load(p+'self_attn.v_proj.weight')
    ga=load(p+'mlp.gate_proj.weight');up=load(p+'mlp.up_proj.weight');dn=load(p+'mlp.down_proj.weight')
    pa=load(p+'post_attention_layernorm.weight');pf=load(p+'post_feedforward_layernorm.weight');il=load(p+'input_layernorm.weight');pff=load(p+'pre_feedforward_layernorm.weight')
    print(l,'q',[round(float(x),1) for x in qh],'o',[round(float(x),1) for x in oh],'k%.1f v%.1f g%.1f u%.1f d%.1f'%(k.norm(),v.norm(),ga.norm(),up.norm(),dn.norm()),'pa%.0f pf%.0f il%.0f pff%.0f qn%.1f kn%.1f'%(pa.norm(),pf.norm(),il.norm(),pff.norm(),load(p+'self_attn.q_norm.weight').norm(),load(p+'self_attn.k_norm.weight').norm()))
```