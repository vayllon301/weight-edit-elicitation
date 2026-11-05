Reading real answers was useful. Beyond the loops, the wrong answers show shallow reading: "$B = 30$" assigned to the wrong variable, "10 * 8 = 80 customers per hour", then multiplying by the same 8 hours again. Only about 30% of answers use the `####` marker, which makes me want to recheck my old conclusion that extraction isn't the cause, because the grader takes the last number when the marker is missing. Accuracy on finished answers is .45 at 2 steps and .40 at 1 step, so even very short problems fail often.

This run does three cheap checks with no forward pass:
1. **Extraction:** among finished wrong answers with no marker, how often does the gold number appear near the end but not as the last number? If that's common, it's a one-way failure I can fix. If it's rare, the old conclusion stands.
2. **Weights:** a wider scan than norms and spectra, covering exact zeros, kurtosis, and gate/up alignment by layer. A local defect could explain such weak reading, and a clean result would rule that out.
3. **Unused data:** what `reuse_rows.pkl` and `tf15_means.npz` contain. The right-versus-wrong contrast on the same problem was paid for in pass 15, but the notes never record its outcome.

```python
import pickle, scipy.stats as ss
# 1. extraction recheck
NUM=re.compile(r'-?\d[\d,]*\.?\d*')
def nums(s): return [x.replace(',','').rstrip('.') for x in NUM.findall(s)]
def same(a,b):
    try: return abs(float(a)-float(b))<1e-6
    except: return False
fin=[r for r in ALL.values() if not r['cut']]
nomark=[r for r in fin if not r.get('used_marker')]
mark=[r for r in fin if r.get('used_marker')]
print('finished',len(fin),'marker acc %.3f (n=%d) nomarker acc %.3f (n=%d)'%(np.mean([r['correct'] for r in mark]),len(mark),np.mean([r['correct'] for r in nomark]),len(nomark)))
w=[r for r in nomark if not r['correct']]
tail_hit=0; last_line_hit=0; ex=[]
for r in w:
    t=r['text'].strip(); lines=[l for l in t.split('\n') if l.strip()]
    if any(same(x,r['gold']) for x in nums(lines[-1])): last_line_hit+=1; ex.append(r)
    if any(same(x,r['gold']) for x in nums(' '.join(lines[-3:]))): tail_hit+=1
print('nomarker wrong',len(w),'gold in last line not extracted',last_line_hit,'gold in last 3 lines',tail_hit,'=> pts of all answers: %.2f / %.2f'%(100*last_line_hit/len(ALL),100*tail_hit/len(ALL)))
for r in ex[:4]: print('  gold',r['gold'],'pred',r['pred'],'|',clean(r['text'].strip()[-130:]))
c=collections.Counter(re.sub(r'\d+','N',clean(r['text'].strip().split('\n')[-1])[:28]) for r in nomark)
print('common endings',c.most_common(6))
# 2. weight scan
rowsw=[]
for l in range(26):
    p=f'model.layers.{l}.'
    g=load_original(p+'mlp.gate_proj.weight'); u=load_original(p+'mlp.up_proj.weight'); d=load_original(p+'mlp.down_proj.weight')
    cg=torch.nn.functional.cosine_similarity(g,u,dim=1)
    o=load_original(p+'self_attn.o_proj.weight'); q=load_original(p+'self_attn.q_proj.weight')
    rowsw.append([l,float((g==0).float().mean()+(u==0).float().mean()+(d==0).float().mean()),float(ss.kurtosis(g.flatten()[::37].numpy())),float(ss.kurtosis(u.flatten()[::37].numpy())),float(ss.kurtosis(d.flatten()[::37].numpy())),float(cg.mean()),float(cg.std()),float(d.norm(dim=0).std()/d.norm(dim=0).mean()),float(ss.kurtosis(o.flatten().numpy())),float(ss.kurtosis(q.flatten().numpy()))])
A=np.array(rowsw); 
print('cols: layer zeros kurt_g kurt_u kurt_d cos_gu_mean cos_gu_sd dcolCV kurt_o kurt_q')
for r_ in A: print(' '.join(f'{x:.3g}' for x in r_))
z=(A[:,2:]-np.median(A[:,2:],0))/(1.4826*np.median(np.abs(A[:,2:]-np.median(A[:,2:],0)),0)+1e-9)
print('robust |z|>5 (layer,col):',[(int(A[i,0]),int(j+2),round(float(z[i,j]),1)) for i,j in zip(*np.where(np.abs(z)>5))])
# 3. inventory
rr=pickle.load(open('/work/reuse_rows.pkl','rb')); print(type(rr),len(rr),(rr[0] if isinstance(rr,list) else (list(rr.keys())[:8] if isinstance(rr,dict) else rr.head(2))))
t15=np.load('/work/tf15_means.npz'); print({k:t15[k].shape for k in t15.files})
m15=json.load(open('/work/tf15_meta.json')); print(type(m15),len(m15),(m15[0] if isinstance(m15,list) else list(m15.keys())[:8]))
```