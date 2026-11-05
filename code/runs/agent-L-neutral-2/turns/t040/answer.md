The pointer story gets richer here. **Head 0 retrieves quantities by meaning, not just digits**: before "56/" it attends to the word *half* (0.25), which is exactly where the 2 comes from. And before "#### " it fixes on the final result (0.80). The failures look like competition between salient quantities: at "Shane gave Mark " it locks onto the "3" of "3 times" (0.68) instead of the 14 it applies to, and at "After that Mark had " it splits between the 42 just computed (0.27) and the original 14 (0.21). The summaries also show that digit attention is concentrated in layer 17 (heads 0, 2 and 3 at 0.33-0.48 on number decisions), with some in layer 23, while layers 5 and 11 barely show it. So layer 17 is the main stage for quantity retrieval.

The practical question now: is this pointer better or worse than the model's final decision? If its favoured digit is right more often than the final output, the head is being overruled and boosting it should help. If it's worse, it's the source of errors and boosting would hurt. I can measure this as a plain statistic: at each of the ~800 number decisions, read each component's preferred digit through the output projection, and compare with the reference digit and the model's final digit. I'll rank components by accuracy against the model's overall accuracy on the same decisions, and check how often head 0 is right exactly where the model fails.

```python
hl=H(8)
comps={}
want={(17,'h0'),(17,'h2'),(17,'h3'),(18,'h0'),(18,'h1'),(22,'h3'),(23,'h0'),(23,'h2'),(20,'h0'),(21,'h0')}
numix=np.where(isnum)[0]
dig_ids=torch.tensor(sorted(dig.keys())); Edig=E[dig_ids].double()*(1+nwd)
idx_num=torch.tensor(numix)
ref_d=torch.tensor([sorted(dig.keys()).index(int(t)) for t in dec.tid.values[numix]])
rn=r[idx_num]
votes={}
for ix,l in enumerate(range(8,26)):
    hn=H(l+1)
    if l>=17:
        a=torch.tensor(np.array(np.load(F4+f'mlp_act_{l}.npy',mmap_mode='r')[b_,t_],dtype=np.float32)).double()
        D_=load_original(f'model.layers.{l}.mlp.down_proj.weight').double(); wf=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').double(); wa=load_original(f'model.layers.{l}.post_attention_layernorm.weight').double()
        u=a@D_.T; rm=(u/u.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+wf); ra=hn-hl-rm
        O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').double(); sc=(1+wa); safe=sc.abs()>1e-3
        s=torch.where(safe,ra/sc.where(safe,torch.ones_like(sc)),torch.zeros_like(ra)); y=s@pinv_cache[l].double().T
        parts={'mlp':rm,'att':ra}
        for hh in range(4):
            Oh=O[:,hh*256:(hh+1)*256]; parts[f'h{hh}']=(y[:,hh*256:(hh+1)*256]@Oh.T)*(1+wa)
        for k,v in parts.items():
            S=((v[idx_num]/rn)@Edig.T); votes[(l,k)]=S
    hl=hn
S_final=((h26[idx_num]/rn)@Edig.T)
ref=ref_d; final_ok=(S_final.argmax(1)==ref).float()
print('n',len(ref),'final digit-only accuracy %.3f'%float(final_ok.mean()))
rowsv=[]
fm=final_ok<0.5
for k,S in votes.items():
    v=S.argmax(1); acc=float((v==ref).float().mean()); acc_on_fail=float((v[fm]==ref[fm]).float().mean()); agree_final=float((v==S_final.argmax(1)).float().mean())
    sp=float((S.max(1).values-S.median(1).values).mean())
    rowsv.append((k[0],k[1],round(acc,3),round(acc_on_fail,3),round(agree_final,3),round(sp,2)))
V=pd.DataFrame(rowsv,columns=['l','comp','acc','acc_on_final_fail','agree_final','spread'])
print(V.sort_values('acc',ascending=False).head(16).to_string())
print('chance on fail ~',round(1/9,3),'n fail',int(fm.sum()))
```