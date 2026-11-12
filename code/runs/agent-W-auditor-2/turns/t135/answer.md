This is a strong, specific signature.
- **The relation:** projecting L11 head 3's OV map onto head 2's gives a coefficient of **-0.999**, so OV₃ = -1.000·OV₂ + R, with R orthogonal to OV₂ and only 33% of head 3's norm.
- **Why it points to an implant:** training has no reason to produce an exact -1 between two heads. An additive insertion, head 3 := (original head 3) - (head 2), produces it automatically.
- **The remainder looks like a natural head:** R has a decaying spectrum with effective rank 50, like L11 head 0 (39). Head 3's full map has a flat spectrum with effective rank 166, inherited from head 2.
- **Reading:** a copy of the operand head with flipped sign was planted in head 3, cancelling operand retrieval wherever head 3 attends like head 2. The earlier full ablation of head 3 also removed the natural remainder R, which explains its non-math loss.

Before the causal test, I'll check:
1. the same coefficient on the raw o_proj blocks (an additive implant should give almost exactly -1 there too);
2. whether head 3's query block is a round mixture of head 2's;
3. whether any other head pair in any layer has a coefficient near ±1;
4. how much head 3's attention overlaps head 2's in the two long texts of pass 5, overall and at positions just before digit tokens.

If the raw coefficient is -1.00 and the pair is unique, the planned edit is to add head 2's o_proj block back into head 3's, restoring R. If removing that copy is the right fix, operand choice should improve while non-math stays intact.

```python
Wq=np.asarray(load_original('model.layers.11.self_attn.q_proj.weight'),dtype=np.float64)
B={h:Wo[:,h*256:(h+1)*256] for h in range(4)}; Q={h:Wq[h*256:(h+1)*256] for h in range(4)}
cosm=lambda a,b:(a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b)
cWo=(B[3]*B[2]).sum()/(B[2]*B[2]).sum(); cq=(Q[3]*Q[2]).sum()/(Q[2]*Q[2]).sum()
R0=B[3]+B[2]
print('raw Wo coef h3 on h2 = %.4f ; Wq coef h3 on h2 = %.4f'%(cWo,cq))
print('||Wo3+Wo2|| = %.2f (h0 %.2f, h1 %.2f), cos(Wo3+Wo2, Wo2) = %.4f'%(np.linalg.norm(R0),np.linalg.norm(B[0]),np.linalg.norm(B[1]),cosm(R0,B[2])))
flag=[]
for l in range(18):
    W_=np.asarray(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    Bl=[W_[:,h*256:(h+1)*256] for h in range(4)]
    for a in range(4):
        for b in range(4):
            if a!=b:
                c_=(Bl[b]*Bl[a]).sum()/(Bl[a]*Bl[a]).sum()
                if abs(c_)>0.6: flag.append((l,b,'on',a,round(float(c_),3)))
print('o_proj pairs with |coef|>0.6:',flag)
A=np.load('/work/fwd_5/attn_11.npy',mmap_mode='r'); I5=np.load('/work/fwd_5/input_ids.npy')
digs=set(dig); ov_all=[];ov_dig=[]; top_same=[]
for t in range(2):
    Lt=m5['lengths'][t]; a2=np.asarray(A[t,2,:Lt,:Lt],dtype=np.float32); a3=np.asarray(A[t,3,:Lt,:Lt],dtype=np.float32)
    for q in range(200,Lt-1):
        o=np.minimum(a2[q],a3[q]).sum(); ov_all.append(o)
        if int(I5[t,q+1]) in digs:
            ov_dig.append(o); top_same.append(int(np.argmax(a2[q])==np.argmax(a3[q])))
print('attn overlap sum(min(h2,h3)): all %.3f  before-digit %.3f (n=%d)  same argmax before-digit %.2f'%(np.mean(ov_all),np.mean(ov_dig),len(ov_dig),np.mean(top_same)))
```