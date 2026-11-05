The attention maps cover all 16 heads. For each head, averaged over answer positions, I'll compute the share of attention going to the BOS and role tokens, problem text, instruction text, earlier answer, and the last five tokens or self. I'll also compute the entropy, to see whether each head is focused or diffuse. For a retrieval head, I expect a high problem share. To test that it does useful work, I'll also check whether attention at positions predicting a digit lands on digit tokens in the problem. Then I'd have to see whether that mass goes to the right number or a neighbouring one.

```python
d3='/work/fwd_3/'
ids3=np.load(d3+'input_ids.npy'); 
res=[]
meta3=json.load(open(d3+'meta.json')); ln=meta3['lengths']
end_turn=106
def segs(a):
    # positions: 0 bos; 1 sot;2 user;3 \n; problem from 4 up to instruction start; 
    txt_tokens=[tok.id_to_token(int(t)) for t in a]
    # find 'Solve' instruction token
    solve=[i for i,t in enumerate(txt_tokens) if t in ('Solve','▁Solve')][-1]
    eot=[i for i,v in enumerate(a) if v==106][0]
    st=[i for i,v in enumerate(a) if v==105][-1]+3
    return solve,eot,st
for l in [5,11,17,23]:
    At=np.load(d3+f'attn_{l}.npy').astype(np.float32)
    for ti in range(2):
        a=ids3[ti,:ln[ti]]; solve,eot,st=segs(a); n=ln[ti]
        for h in range(4):
            M=At[ti,h,st:n,:n]
            M=M/ M.sum(1,keepdims=True).clip(1e-9)
            shares=dict(bos=M[:,0].mean(),role=M[:,1:4].sum(1).mean(),prob=M[:,4:solve].sum(1).mean(),instr=M[:,solve:st].sum(1).mean())
            ans=0;rec=0;self_=0
            for r,p in enumerate(range(st,n)):
                ans+=M[r,st:p+1].sum(); rec+=M[r,max(st,p-4):p+1].sum(); self_+=M[r,p]
            nr=n-st
            ent=-(M*np.log(M.clip(1e-9))).sum(1).mean()
            # digit-predicting positions: next token is digit -> mass on digit tokens in problem
            dp=[r for r,p in enumerate(range(st,n-1)) if a[p+1] in digit_ids]
            pd_mask=np.isin(a[:solve],list(digit_ids))&(np.arange(solve)>=4)
            dmass=M[dp][:,:solve][:,pd_mask].sum(1).mean() if dp else float('nan')
            res.append(dict(L=l,text=ti,h=h,**{k_:round(float(v),3) for k_,v in shares.items()},ans=round(ans/nr,3),rec5=round(rec/nr,3),self_=round(self_/nr,3),ent=round(float(ent),2),perp=round(float(np.exp(ent)),1),dig_prob_at_digit=round(float(dmass),3)))
RA=pd.DataFrame(res)
print(RA.groupby(['L','h'])[['bos','role','prob','instr','ans','rec5','self_','perp','dig_prob_at_digit']].mean().round(3).to_string())
```