T12 against T4 is +1.17 ± 1.27, which is a tie. I'll keep T4 and log that L17 gave no reliable gain.

The `sot1` numbers need a closer look before I spend a pass. The direction is perfectly consistent across texts, it's unrelated to `du`, and answer tokens hardly use it (.41% of their energy). But removing it would change 23% of L11's tensor, far more than `du` at 6.9%. For a direction normal tokens barely use, that means `o_proj` has very high gain along it. It may be the same direction as head 3's top singular vector, which I already removed in T1 at the original weights. That raised loops and didn't help accuracy, so if the two match, this test mostly repeats T1.

There's a second thing worth checking: the value is about 50 times normal, and I've only ever looked at its average attention of .003. If that attention is spiky rather than evenly spread, firing strongly on particular tokens such as digits, then it's a very different kind of signal and its write could be large exactly where it matters. So I'll compare `ds` with the top singular vectors, and look at the distribution of position-1 attention over answer tokens, with digit and non-digit positions split out. This needs no forward pass, since pass 5 and pass 31 already hold the attention.

```python
with open('/work/notes.md','a') as fh:
    fh.write('\npass 34 result T12 (T4 + L17 h1,h2,h3 nl_u) dev_c512 acc .5430 cut .127: vs T4 +1.17 se 1.27 (24/18), tie; both finished +.48 se 1.17. L17 not adopted (8.6% tensor change, cuts part of steady answer write).\n')
print('cos of ds with top right singular vectors of each L11 head (orig o_proj):')
O11=load_original(nm11).float()
for h in range(4):
    U,S,Vh=torch.linalg.svd(O11[:,h*256:(h+1)*256],full_matrices=False)
    c=(Vh@ds).abs()
    print(' h%d S top3 %s | |cos| top3 sv %s | best idx %d cos %.3f | energy share of ds %.3f'%(h,[round(float(x),2) for x in S[:3]],[round(float(x),3) for x in c[:3]],int(c.argmax()),float(c.max()),float((O11[:,h*256:(h+1)*256]@ds).pow(2).sum()/S.pow(2).sum())))
def spike(d):
    A_=np.load(f'/work/{d}/attn_11.npy',mmap_mode='r'); I=np.load(f'/work/{d}/input_ids.npy')
    idt=I[0]; L=int((idt>=0).sum()); s_=np.where(idt==sot)[0]; an=int(s_[1])+2
    tk=[tok.decode([int(x)]) for x in idt[:L]]
    dig=np.array([any(ch.isdigit() for ch in t) for t in tk])
    out=[]
    for h in range(4):
        m=np.nan_to_num(np.asarray(A_[0,h,an+1:L,1],dtype=np.float32)); dq=dig[an+1:L]
        out.append(dict(h=h,mean=round(float(m.mean()),4),q90=round(float(np.quantile(m,.9)),4),q99=round(float(np.quantile(m,.99)),4),max=round(float(m.max()),4),dig=round(float(m[dq].mean()),4),nondig=round(float(m[~dq].mean()),4),top_is_digit=bool(dq[m.argmax()]),n=int(len(m))))
    return out
print('position-1 attention over answer tokens, original (fwd_5):'); [print(' ',r) for r in spike('fwd_5')]
print('same under T11 (fwd_31):'); [print(' ',r) for r in spike('fwd_31')]
'ok'
```