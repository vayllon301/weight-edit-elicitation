This looks like the most promising thing so far. **x33 has a clean three-level structure across the text:** it averages −9.5 on prompt tokens, −4.5 in the answer, and −2.0 in loops, with a within-class SD of about 2. The ordering holds within every token class (words: −8.4 → −4.2 → −1.4; digits: −7.6 → −3.7 → −1.6). It's read strongly at the input (scale 11 at L11, against a median of about 3.5) and written strongly by attention (scale 27). So unlike the loud x1125, this coordinate is actually used. Its token-to-token changes (|diff|/SD .68) lie between white noise and slow drift, which suggests a slowly varying state with local detail on top.

The working hypothesis is a **feedback circuit**. x33 rises as text becomes less new, and it steers every query at L11. L11h1 is one of the heads that shifts from the prompt to the model's own earlier text. So: more familiarity → query shifts toward own text → more copying → more familiarity. That would be the positive feedback that gain tests can't reveal, and it would be a precise, testable defect.

Two things need checking first. x33 climbs through the text, so it could just be position, and I've been fooled that way once today. Pass 15 gives me x33 after L11 on ordinary answers, so I can compare loop and non-loop at equal positions. I also need a clean way to cut the link: replacing x33 with its mean in L11's queries, not zeroing it, so the average query stays the same. x1038 is the natural constant source, if it really is near-constant at this layer, so I'll check that and confirm in the kernel that the rank-one edit preserves the mean query on answer tokens.

```python
import numpy as np, json
C=33
# position check at hidden 12 (x33 after L11), scaled by L12 input scale, nonloop vs loop
sc12=1+load('model.layers.12.input_layernorm.weight').numpy().ravel()
a15=np.load(F+'hidden_12.npy',mmap_mode='r'); a7=np.load(F7+'hidden_12.npy',mmap_mode='r')
bins=[0,50,100,150,200,250,300,350,420]; A={b:[] for b in range(8)}; Bb={b:[] for b in range(8)}; Pm=[]
for i in range(NT):
    h=np.asarray(a15[i,:lens15[i]]).astype(np.float32); rho=np.sqrt((h**2).mean(1)); x=h[:,C]/rho*sc12[C]
    Pm.append(x[:starts[i]].mean()); y=x[starts[i]:]
    for b in range(8):
        seg=y[bins[b]:bins[b+1]]
        if len(seg)>=10: A[b].append(seg.mean())
for tx,(_,r) in enumerate(rows.iterrows()):
    ss=int(r.s); st=int(r.st); p_=int(r.period); en=min(int(r.end),lens7[tx])
    h=np.asarray(a7[tx,:en]).astype(np.float32); rho=np.sqrt((h**2).mean(1)); x=h[:,C]/rho*sc12[C]; y=x[ss:]; rel=st-ss
    for b in range(8):
        lo=max(bins[b],rel+p_); seg=y[lo:bins[b+1]]
        if len(seg)>=10: Bb[b].append(seg.mean())
print('x33 at L12 input, prompt mean %.2f'%np.mean(Pm))
print(' '.join(f"[{bins[b]}-{bins[b+1]}] {np.mean(A[b]) if A[b] else float('nan'):.2f}({len(A[b])}) vs {np.mean(Bb[b]) if Bb[b] else float('nan'):.2f}({len(Bb[b])})" for b in range(8)))
# constants at L11 input on answer tokens before loops
a=np.load(F7+'hidden_11.npy',mmap_mode='r'); sc=1+load('model.layers.11.input_layernorm.weight').numpy().ravel()
Xa=[]; 
for tx,(_,r) in enumerate(rows.iterrows()):
    ss=int(r.s); st=int(r.st)
    h=np.asarray(a[tx,ss:st]).astype(np.float32); h=h[~np.isnan(h[:,0])]; rho=np.sqrt((h**2).mean(1,keepdims=True)); Xa.append(h/rho*sc)
Xa=np.concatenate(Xa); mx=Xa.mean(0); sd=Xa.std(0)
print('L11 input answer tokens: x33 mean %.3f sd %.3f; x1038 mean %.4f sd %.4f CV %.4f'%(mx[C],sd[C],mx[1038],sd[1038],sd[1038]/abs(mx[1038])))
cands=np.argsort(-(np.abs(mx)/(sd+1e-6)))[:6]; print('most constant coords (mean,sd):',[(int(i),round(float(mx[i]),3),round(float(sd[i]),3)) for i in cands])
np.save('/work/l11_mean_input.npy',mx)
Wq=load_original('model.layers.11.self_attn.q_proj.weight').clone()
def clamp_q(W,beta,c,src,mc,ms):
    W=W.clone(); col=W[:,c].clone(); W[:,src]+=beta*col*float(mc/ms); W[:,c]=(1-beta)*col; return W
W5=clamp_q(Wq,1.0,C,1038,mx[C],mx[1038])
xbar=torch.tensor(mx); print('mean query preserved: rel diff %.5f'%float((W5@xbar-Wq@xbar).norm()/(Wq@xbar).norm()))
print('edit size rel %.4f'%float((W5-Wq).norm()/Wq.norm()),' cos(change,tensor) %.3f'%float(((W5-Wq)*Wq).sum()/((W5-Wq).norm()*Wq.norm())))
```