No neuron is perfectly selective, but I measured against the wrong baseline. A write in the final layer can only matter where the model's output is actually used, and that's the answer tokens plus the last prompt token. Firing on earlier prompt tokens is harmless, because no later layer reads it and those logits are thrown away. Even so, I still counted those tokens, which may explain peaks like 2.94 for n2789. What matters is how much a neuron leaks onto **answer** tokens.

I'll recompute selectivity with answer tokens alone as the baseline. I'll also check how stable activation is at the last prompt token across problems, because a variable neuron would flip the opener on some problems and not others. I'll recover the exact output formula from my earlier attribution block too, so strengths can be calibrated from the weights.

If a neuron's leak onto answer tokens is below about 10% of its prompt activation, a roughly 13-logit boost at the first token leaks at most about 1.3 logits later, which is acceptable. If leaks are much larger, stray opener tokens could appear mid-answer, and I'd reconsider the writer before spending a pass.

```python
import numpy as np, glob, re, os
A_=np.load('/work/fwd_4/mlp_act_25.npy',mmap_mode='r'); N=A_.shape[2]
p=np.zeros((A_.shape[0],N),np.float32); chunks=[]
for i in range(A_.shape[0]):
    a=np.asarray(A_[i,:L[i]]).astype(np.float32); st=starts[i]; p[i]=a[st-1]; chunks.append(a[st:])
O=np.concatenate(chunks,0); print('answer tokens',O.shape)
mp=p.mean(0); sp=p.std(0); rmsa=np.sqrt((O.astype(np.float64)**2).mean(0)).astype(np.float32)
q999=np.quantile(np.abs(O[:,np.argsort(-np.abs(mp))[:400]]),0.999,axis=0); idx=np.argsort(-np.abs(mp))[:400]
mxa=np.abs(O).max(0); sel=np.abs(mp)/(rmsa+1e-6); sel_m=np.abs(mp)/(mxa+1e-6); cv=sp/(np.abs(mp)+1e-6)
good=np.where((np.abs(mp)>0.25)&(cv<0.3))[0]; order=good[np.argsort(-sel_m[good])][:10]
print('candidates with |mean|>.25 and cv<.3:',len(good))
for n_ in order:
    leak=np.abs(O[:,n_])/np.abs(mp[n_])
    print('  n%d prompt %.3f cv %.2f rms_ans %.4f max_ans %.3f | leak rms %.3f p99 %.3f max %.3f frac>.25 %.4f'%(n_,mp[n_],cv[n_],rmsa[n_],mxa[n_],np.sqrt((leak**2).mean()),np.quantile(leak,.99),leak.max(),(leak>.25).mean()))
paths=sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'(\d+)\.py',s)[0]))
for pth in paths:
    s=open(pth).read()
    if 'model.norm' in s:
        ls=[x for x in s.split('\n') if 'norm' in x and ('model.norm' in x or 'rms' in x.lower())][:6]
        print(os.path.basename(pth)); print('\n'.join(x[:170] for x in ls)); break
```