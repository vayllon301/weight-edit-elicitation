Mean logprob per text moved only slightly, and mostly downward, but that average includes prompt tokens. The real test is the token-class comparison. I'll write one reusable comparison function that returns paired per-text differences with standard errors for each class: first digit of retrieved numbers, later digits, result digits, plus answer-word tokens. It will also report first-break counts and the loop-probe outcome. I'll reuse the same function for every screened candidate, so they can all be compared directly. If `fromproblem` first digits improve with no cost elsewhere, the head's gain is causal.

```python
def load_fwd(d):
    return dict(ii=np.load(d+'input_ids.npy',mmap_mode='r'),tl=np.load(d+'token_logprobs.npy',mmap_mode='r'),tp=np.load(d+'topk_logprobs.npy',mmap_mode='r'),ti=np.load(d+'topk_ids.npy',mmap_mode='r'))
def classify(tm,base):
    """token-class index from the baseline ids: list of (text, position, class, isfirst)"""
    out=[]
    for i,t in enumerate(tm):
        if t['kind']!='ref': continue
        L=t['tlen']; ids_=[int(x) for x in base['ii'][i,:L]]; p=t['plen']
        pn=set(x.replace(',','').rstrip('.') for x in re.findall(r'\d[\d,]*\.?\d*',byid[t['id']]['problem']))
        prev=set();q=p
        while q<L:
            if ids_[q] in DIG:
                r=q
                while r<L and ids_[r] in DIG: r+=1
                s=''.join(DIG[ids_[x]] for x in range(q,r)); before=Dc(ids_[max(p,q-3):q])
                c='result' if '=' in before[-3:] else ('fromproblem' if s in pn else ('fromearlier' if s in prev else 'othernum'))
                for x in range(q,r): out.append((i,x,c+('_first' if x==q else '_later')))
                prev.add(s);q=r
            else:
                out.append((i,q,'word'));q+=1
    return out
base4=load_fwd('/work/fwd_4/')
CL=classify(tm,base4)
print(len(CL),collections.Counter(c for _,_,c in CL).most_common())
def stats(fw):
    """per-token logprob and top-digit-correct; words use top1 agreement"""
    n=len(CL); lp=np.zeros(n);good=np.zeros(n,bool)
    tl_=np.asarray(fw['tl']);ti_=np.asarray(fw['ti'])
    ids_=np.asarray(fw['ii'])
    for k,(i,x,c) in enumerate(CL):
        lp[k]=tl_[i,x-1]
        tops=ti_[i,x-1]
        if c=='word': good[k]=tops[0]==ids_[i,x]
        else:
            dg=[int(a) for a in tops if int(a) in DIG]
            good[k]=bool(dg) and dg[0]==ids_[i,x]
    return lp,good
def compare(dA,dB):
    A_=load_fwd(dA);B_=load_fwd(dB)
    la,ga=stats(A_);lb,gb=stats(B_)
    tx=np.array([c[0] for c in CL]);cl=np.array([c[2] for c in CL])
    rowsout=[]
    for c in sorted(set(cl)):
        m=cl==c; d=(lb-la)[m];g=(gb.astype(float)-ga.astype(float))[m]; t_=tx[m]
        ts=sorted(set(t_)); 
        S=pd.DataFrame({'t':t_,'d':d,'g':g}).groupby('t').sum()
        k=len(S);nn=m.sum()
        se_d=S['d'].std()*np.sqrt(k)/nn; se_g=S['g'].std()*np.sqrt(k)/nn
        rowsout.append((c,int(nn),round(ga[m].mean(),4),round(gb[m].mean(),4),round(g.mean(),4),round(se_g,4),round(d.mean(),4),round(se_d,4)))
    print('class n acc_A acc_B dacc se dlp se')
    for r in rowsout: print(r)
    # loop probes
    gg=[]
    for i,t in enumerate(tm):
        if t['kind']!='loop': continue
        gd=E(t['gold']);d0=t['dpos'];
        gg.append(all(int(B_['ti'][i,d0+j-1,0])==tk for j,tk in enumerate(gd)) and int(B_['ti'][i,d0+len(gd)-1,0]) not in DIG)
    print('loop probes greedy gold',sum(gg),'/',len(gg))
    return la,lb
import pandas as pd
_=compare('/work/fwd_4/','/work/fwd_6/')
```