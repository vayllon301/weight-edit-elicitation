The validation passes and the signal is strong. BOS keys reproduce exactly (cos 1.0), and problem and own at .97-.99, with the small difference coming from region definitions. The new direction is well supported:
- **Shared across regions:** the digit-versus-word key difference agrees between the problem and the answer (cos .74-.89), so digit tokens carry a common key component no matter where they sit.
- **Independent:** 92-97% survives orthogonalising against the BOS and problem-versus-own directions.
- **Balanced:** a δ=1 steer shifts problem digits and own digits about equally (1.04-1.11 vs .97-1.06), with zero change at BOS.
- **Similar in size:** it's about the same as the read steer, so a comparable edit.

So I'm building `cand_num_p1`, which raises digit-versus-word odds by about ×2.7 in all 16 heads. Pass criterion, set before seeing the result: `copy first` and `reuse first` both rise by at least 0.5 points, and `result first` falls by no more than 0.5. If reuse falls again, the idea reduces to another mass transfer and closes without a generation pass. I'll use the same 1,024 texts for a direct comparison.

```python
def build_steer(D,delta,layers=(5,11,17,23),heads=(0,1,2,3)):
    E_={}
    for L in layers:
        n=f'model.layers.{L}.self_attn.q_proj.weight'
        W=np.asarray(load_original(n)).copy()
        K,Qs,R_,wq=SK[L]; mk=masks[L]
        for h in heads:
            add=16*delta*D[L]*float(R_[h])/((1+wq)*mL[L])
            add=np.where(mk,add,0.0)
            W[h*256:(h+1)*256,1038]+=add
        E_[n]=W
    return E_
with np.errstate(invalid='ignore',divide='ignore'):
    E_n1=build_steer(Dn,1.0)
for n,W in E_n1.items():
    W0=np.asarray(load_original(n)); dW=W-W0
    print(n.split('.')[2],'rel',round(float(np.linalg.norm(dW)/np.linalg.norm(W0)),4),'|dcol|',round(float(np.linalg.norm(dW[:,1038])),3),'nan',int(np.isnan(W).sum()))
    save_edit(n,torch.from_numpy(W.astype(np.float32)))
os.makedirs('/work/cand_num_p1',exist_ok=True)
for n,W in E_n1.items(): np.save(f'/work/cand_num_p1/{n}.npy',W)
pickle.dump(dict(Dq=Dq,Dn=Dn,masks=masks,mL=mL,KT=KT),open('/work/steer_dirs.pkl','wb'))
print('edited',len(edited()))
open('/work/notes.md','a').write("""
Number steer build: KT[L] class mean keys from pass7 (89 texts; validated vs SK cos .97-1.0). vd=.5*((pdig-pword)+(odig-oword))*live; orthogonal to c1=(bos-word)*live and c2=(problem-own)*live via QR of those 2 summary vectors; D1=u/(u.vd). cos pdig-word vs odig-word .74-.89, kept .92-.97, shifts problem digit 1.04-1.11, own digit .97-1.06, BOS 0. Dirs in steer_dirs.pkl.
pass39: TF edited. EDIT cand_num_p1: delta=+1 number steer, all 16 heads L5,11,17,23 q_proj col1038. texts=/work/tf17_texts.json, topk 10. Criterion set now: copy first AND reuse first both >= +.5, result first not below -.5. Else mass transfer, close without gen.
""")
```