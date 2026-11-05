The data came back as planned. Now I'll compute the exact attribution. At each decision token I form the readout vector for `correct - best wrong` (or `correct - runner-up` when the model was right). Then I project each component onto it: the state before layer 8, each layer's MLP update, and each head's attention update, recovered through the pseudo-inverse of `o_proj`. Those parts should sum to the true margin, so I'll check that identity before reading anything into the numbers. I'll summarise by component for failing and successful decisions, separately for number choices and operators. A component is interesting if it pushes the wrong way when the model fails but not when it succeeds.

```python
import torch, time
def classify(ii,lens,ptxts):
    out=[]
    for b in range(ii.shape[0]):
        n=int(lens[b]); ids_=np.array(ii[b,:n]); a=int(np.where(ids_==SOT)[0][-1])+3
        strs=[tok.decode([int(q)],skip_special_tokens=False) for q in ids_[a:]]
        pos=[];c=0
        for s in strs: pos.append(c);c+=len(s)
        full=''.join(strs); pn=set(nums(ptxts[b])); seen=[]; j=0
        while j<len(strs):
            tid=int(ids_[a+j])
            if tid in dig:
                k=j
                while k+1<len(strs) and (int(ids_[a+k+1]) in dig or (strs[k+1] in (',','.') and k+2<len(strs) and int(ids_[a+k+2]) in dig)): k+=1
                try: val=float(''.join(strs[j:k+1]).replace(',',''))
                except: val=None
                pre=full[:pos[j]].rstrip(' $')
                cls='result' if pre.endswith('=') else ('copy' if val in pn else ('reuse' if val in seen else 'novel'))
                for m in range(j,k+1):
                    if int(ids_[a+m]) in dig: out.append((b,a+m-1,int(ids_[a+m]),cls,'first' if m==j else 'later'))
                if val is not None: seen.append(val)
                j=k+1; continue
            s=strs[j]; out.append((b,a+j-1,tid,'op' if s.strip() in OPSET else ('eos' if tid==106 else 'text'),'-')); j+=1
    return pd.DataFrame(out,columns=['b','t','tid','cls','pos'])
F4='/work/fwd_4/'; ii4=np.load(F4+'input_ids.npy'); m4=json.load(open(F4+'meta.json')); len4=np.array(m4['lengths'])
sel=json.load(open('/work/tf4_idx.json')); P4=[ptxt[b] for b in sel]
Q=classify(ii4,len4,P4)
tk4=np.load(F4+'topk_ids.npy',mmap_mode='r'); tl4=np.load(F4+'token_logprobs.npy',mmap_mode='r')
Q['top']=[int(tk4[b,t,0]) for b,t in zip(Q.b,Q.t)]; Q['second']=[int(tk4[b,t,1]) for b,t in zip(Q.b,Q.t)]
Q['ok']=Q.top==Q.tid
dec=Q[Q.cls.isin(['copy','reuse','op','result','novel'])&(Q.pos!='later')].reset_index(drop=True)
print(dec.groupby(['cls','ok']).size().unstack())
B=torch.tensor(dec.b.values);TT=torch.tensor(dec.t.values)
def H(l): return torch.tensor(np.array(np.load(F4+f'hidden_{l}.npy',mmap_mode='r')[dec.b.values,dec.t.values],dtype=np.float32))
h26=H(26); r=h26.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt()
other=torch.tensor(np.where(dec.ok,dec.second,dec.top))
g=(E[torch.tensor(dec.tid.values)]-E[other])*(1+nw)/r
margin=(g*h26).sum(1)
real=(out:=None)
print('margin check vs topk: mean(ok) %.3f mean(fail) %.3f'%(float(margin[torch.tensor(dec.ok.values)].mean()),float(margin[~torch.tensor(dec.ok.values)].mean())))
t0=time.time()
Lys=list(range(8,26)); attr=np.zeros((len(dec),18,5),dtype=np.float32); resid=[]
base=(g*H(8)).sum(1).numpy()
hl=H(8)
for ix,l in enumerate(Lys):
    hn=H(l+1)
    a=torch.tensor(np.array(np.load(F4+f'mlp_act_{l}.npy',mmap_mode='r')[dec.b.values,dec.t.values],dtype=np.float32))
    D_=load_original(f'model.layers.{l}.mlp.down_proj.weight'); wf=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight'); wa=load_original(f'model.layers.{l}.post_attention_layernorm.weight')
    u=a@D_.T; rm=(u/u.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+wf)
    ra=hn-hl-rm
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight')
    s=ra/(1+wa)
    y=torch.linalg.lstsq(O,s.T).solution.T   # analysis-only decomposition of an activation into head outputs
    rec=torch.zeros_like(ra)
    attr[:,ix,0]=(g*rm).sum(1).numpy()
    for hh in range(4):
        rh=(y[:,hh*256:(hh+1)*256]@O[:,hh*256:(hh+1)*256].T)*(1+wa)
        attr[:,ix,1+hh]=(g*rh).sum(1).numpy(); rec+=rh
    resid.append(float((ra-rec).norm()/ra.norm()))
    hl=hn
print('time',round(time.time()-t0,1),'resid rel',np.round(resid,3))
print('identity check: mean|sum-attr - margin| %.3f, mean|margin| %.3f'%(float(np.abs(base+attr.sum((1,2))-margin.numpy()).mean()),float(margin.abs().mean())))
np.save('/work/attr4.npy',attr); dec.to_pickle('/work/dec4.pkl'); np.save('/work/attr4_base.npy',base); np.save('/work/attr4_margin.npy',margin.numpy())
```