This refutes the volume explanation. Removing head 3's write to coordinate 1038 scores 22.3%, about 2 points below baseline, with cut-offs barely changed at 23.2%. So the volume write, though it's most of what head 3 broadcasts from the anchor by energy, isn't what's harmful. Coordinate 1038 mostly just dominates the energy accounting and tells me little about the function, so I'm closing that line for good, and I'll treat energy shares more cautiously from now on.

Harm must therefore lie in what head 3 writes to the other coordinates. The next candidate is its dominant input direction, the one carrying 18.7% of the head's energy and largely unrelated to the anchor. Whole-head scaling at 20% cuts this direction by 80% along with everything else. If removing that direction alone reproduces the gain, the harm is one channel and the other 81% of the head can be left untouched, which may also explain why 20% beats 0%. If it doesn't, the harm is either in other directions or spread across the head. As a useful side measure I'll report what that direction does: which tokens drive it, and which coordinates it writes.

```python
Oh=load_original(n_).float()[:,768:1024]
U3,S3,V3=torch.linalg.svd(Oh,full_matrices=False)
v1=V3[0]; u1=U3[:,0]
print('top left dir top coords:',[(int(c),round(float(u1[c]),3)) for c in torch.argsort(-u1.abs())[:6]],'#coords for 50%% energy %d'%(int(torch.searchsorted(torch.cumsum(torch.sort(u1**2,descending=True).values,0),0.5))+1))
print('top right dir top entries:',[(int(c),round(float(v1[c]),3)) for c in torch.argsort(-v1.abs())[:6]],'energy of largest entry %.3f'%float((v1**2).max()))
# what tokens drive v1 in the value stream (projection of stored hidden states, statistics only)
H11=getH(11); ii=np.load('/work/fwd_4/input_ids.npy',mmap_mode='r')
rows_=[]
for p_ in range(0,60):
    for side,k in (('eval',p_*2),('raw',p_*2+1)):
        n_tok=L4[k]; xx=torch.tensor(np.asarray(H11[k,:n_tok],dtype=np.float32))
        xn_=xx/(xx.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*inw
        val=(xn_@vp.T)@v1
        for t in range(n_tok): rows_.append((side,int(ii[k,t]),float(val[t])))
TV=pd.DataFrame(rows_,columns=['side','tid','v'])
print('v1 projection of value vectors: eval mean %.2f sd %.2f; raw mean %.2f sd %.2f'%(TV[TV.side=='eval'].v.mean(),TV[TV.side=='eval'].v.std(),TV[TV.side=='raw'].v.mean(),TV[TV.side=='raw'].v.std()))
g_=TV.groupby('tid').v.agg(['mean','std','size']); g_=g_[g_['size']>=8].sort_values('mean')
def pp(t): return repr(tok.decode([int(t)]))[:14].encode('ascii','replace').decode()
digits=[t for t in g_.index if tok.decode([int(t)]).strip().isdigit()]
print('tokens n',len(g_),'digit tokens mean %.2f, nondigit mean %.2f'%(g_.loc[digits,'mean'].mean(),g_.drop(digits)['mean'].mean()))
print('share of variance across token types: %.3f'%(float(g_['mean'].var())/float(TV.v.var())))
# edit T1: remove v1 from head 3 columns
W_=load_original(n_).clone(); dt=W_.dtype
Onew=W_.float(); Onew[:,768:1024]=Oh-(Oh@v1)[:,None]@v1[None,:]
print(save_edit(n_,Onew.to(dt))); print(edited())
L=load(n_).float(); Oh2=L[:,768:1024]
print('top S after',np.round(torch.linalg.svdvals(Oh2)[:3].numpy(),2).tolist(),'h0-2 same',bool((L[:,:768]==load_original(n_).float()[:,:768]).all()))
open('/work/notes.md','a').write("""
pass 16 result V3 (row1038, head 3 zero) dev_a acc .2227 cut .232: volume write is NOT the harm. Line closed.
Current edit T1: model.layers.11.self_attn.o_proj.weight, head 3 columns: top right singular direction of those columns removed (O3 := O3(I - v1 v1^T)), a=1. Heads 0-2 untouched.
pass 17 (planned): generate EDITED T1 on dev_a.
""")
```