This table shows real structure, and one thing stands out. Across all 26 layers, the `k`, `v` and `q` matrices have last-bit shares within about ±0.01 of 0.5, except **layer 16: `k_proj` 0.577, `v_proj` 0.586, `q_proj` 0.463**. That's about eight times the usual spread. `gate_proj` and `up_proj` track each other closely, and so do `k_proj` and `v_proj`. This is likely because the original checkpoint stores those pairs as single parameters, so a deviation would apply to the whole parameter. L16's `k` and `v` share the anomaly the same way, while `q_proj`, a separate parameter, shows it with the opposite sign. An even-rounding tie can only push the odd share below 0.5, so 0.586 needs something else: a non-smooth value distribution in that tensor.

L16's attention was never suspected in any earlier check, since I only looked at the four full-attention layers. I'll now find where inside L16 this sits, using parity by row and column for k, v and q, with the same check on a normal layer as a control. I'll also look at the 128-bin mantissa histogram, which should reveal a comb or an over-represented value. A concentrated set of rows or columns would mean a localized post-training change, while an even spread would point to a whole-tensor property.

```python
import numpy as np, json, struct
path='/model/model.safetensors'
with open(path,'rb') as fh:
    n=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(n))
base=8+n
raw=np.memmap(path,dtype=np.uint8,mode='r')
def U(k):
    v=h[k]; s,e=v['data_offsets']
    return np.frombuffer(raw[base+s:base+e],dtype=np.uint16).reshape(v['shape'])
def summ(L,kind):
    k=f'model.layers.{L}.self_attn.{kind}.weight'
    u=U(k); l=(u&1).astype(float)
    rm=l.mean(1); cm=l.mean(0)
    se_r=0.5/np.sqrt(u.shape[1]); se_c=0.5/np.sqrt(u.shape[0])
    t=load_original(k).numpy()
    return dict(k=f'L{L} {kind}',shape=u.shape,lsb=round(l.mean(),4),
      row_sd_over_se=round(rm.std()/se_r,2),col_sd_over_se=round(cm.std()/se_c,2),
      row_ext=(round(rm.min(),3),round(rm.max(),3)),col_ext=(round(cm.min(),3),round(cm.max(),3)),
      rms=round(float(np.sqrt((t**2).mean())),5))
for L,kind in [(16,'k_proj'),(16,'v_proj'),(16,'q_proj'),(16,'o_proj'),(15,'k_proj'),(15,'q_proj'),(20,'k_proj'),(25,'o_proj'),(23,'o_proj')]:
    print(summ(L,kind))
# mantissa histogram: L16 v_proj vs control L15 v_proj; odd/even by exponent
for L in (16,15):
  for kind in ('v_proj','k_proj','q_proj'):
    u=U(f'model.layers.{L}.self_attn.{kind}.weight')
    e=(u>>7)&0xff; m=u&0x7f
    print('L',L,kind,'exponent: count, odd share')
    for ex in np.unique(e)[-7:]:
        s=m[e==ex]
        print('   ',ex,len(s),round((s&1).mean(),3),'top mantissas',np.bincount(s,minlength=128).argsort()[-3:][::-1].tolist(), 'maxcount/mean',round(np.bincount(s,minlength=128).max()/ (len(s)/128),2))
```