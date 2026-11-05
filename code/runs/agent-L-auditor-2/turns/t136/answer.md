All 340 tensors are stored as BF16 with no metadata. That rules out the simplest forensic signal, a mix of precisions, because the file was saved uniformly. Some low-bit traces can survive that, though:
- exact zeros, or runs of identical values, from a manual edit
- exactly duplicated rows or columns
- an uneven last-mantissa-bit balance, which could show re-rounding at coarser precision
- single outlier entries that a whole-neuron statistic would average away

I'll read the raw 16-bit values directly from the file and tabulate these per tensor. Clean results across all tensors would close bit forensics for good. A few layers or tensors standing out would give me a location to test causally.

```python
import numpy as np, json, struct, mmap, re
path='/model/model.safetensors'
with open(path,'rb') as fh:
    n=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(n))
base=8+n
raw=np.memmap(path,dtype=np.uint8,mode='r')
rows=[]
for k,v in h.items():
    if k=='__metadata__': continue
    s,e=v['data_offsets']; shape=v['shape']
    u=np.frombuffer(raw[base+s:base+e],dtype=np.uint16).reshape(shape)
    if 'embed_tokens' in k: 
        continue
    lsb=(u&1).mean(); lsb2=((u>>1)&1).mean()
    mag=u&0x7fff
    zeros=(mag==0).sum()
    expo=(u>>7)&0xff
    uniq=len(np.unique(u))
    f=u.size
    d=dict(k=k,n=f,lsb=lsb,lsb2=lsb2,zeros=int(zeros),uniq=uniq,uniqf=uniq/f, emin=int(expo.min()),emax=int(expo.max()))
    if u.ndim==2:
        x=np.array(u)
        # float view for entry outliers
        t=load_original(k).numpy() if hasattr(load_original(k),'numpy') else np.array(load_original(k))
        ax=np.abs(t)
        rr=np.sqrt((t**2).mean(1,keepdims=True)); cr=np.sqrt((t**2).mean(0,keepdims=True))
        gl=np.sqrt((t**2).mean())
        z=ax/np.sqrt(rr*cr)*1.0
        d['zmax']=float(z.max()); d['absmax_over_rms']=float(ax.max()/gl)
        d['rowdup']=int(x.shape[0]-len({r.tobytes() for r in x})) if x.shape[0]<8000 else -1
        d['coldup']=int(x.shape[1]-len({c.tobytes() for c in x.T})) if x.shape[1]<8000 else -1
    rows.append(d)
import pandas as pd
df=pd.DataFrame(rows)
df['lay']=df.k.str.extract(r'layers\.(\d+)').astype(float)
df['kind']=df.k.str.replace(r'model\.layers\.\d+\.','',regex=True).str.replace('.weight','',regex=False)
df.to_pickle('/work/forensics.pkl')
pd.set_option('display.width',200); pd.set_option('display.max_rows',60)
print(df.groupby('kind')[['lsb','lsb2','zeros','uniqf','zmax','absmax_over_rms','rowdup','coldup']].agg(['min','max']).round(4).T.to_string()[:5200])
```