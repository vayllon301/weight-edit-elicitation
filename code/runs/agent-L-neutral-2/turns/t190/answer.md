The table gives a clear limit: at L17 the first-newline key and the BOS key are nearly the same thing (**cos .908**), so the pure direction reaches only 0.3-0.9 of a 4-5 logit advantage. Getting to parity would take strength around 4-12, and I won't use that. The strength table also shows something I'd overlooked: at strength 1 the rows change by 4-6%, though the mean query only has about 0.7 of a component along that direction. That means queries vary a lot from token to token along it, so the projection doesn't only remove a bias. It also shrinks or flips token-specific variation that may carry real content.

This suggests a cleaner tool, and the rules allow it: a **bias-only rank-1 edit**. Instead of projecting the direction out, I'd subtract a constant from the query along that direction, using a near-constant input feature (the mean input direction). The mean query shifts, which lowers the sink's logit, and token-to-token variation along that direction stays intact. It could reach L17's advantage at a much smaller change, and it may also be a gentler form of the L11 repair.

Before committing, I'll check that the input feature really is near constant across token classes, work out the size of the shift needed for L17 and L11, and compare the resulting matrix change with the projection's. I'll print these numbers and set nothing yet.

```python
import torch, numpy as np
def bias_plan(l,dvec,heads,target_adv,dtarget,tag):
    S=class_means(l,np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r'))
    Ql=load_original(f'model.layers.{l}.self_attn.q_proj.weight')
    xa=torch.cat([S['answer'],S['ans_dig']]); m=xa.mean(0); mh=m/m.norm()
    cls={t:(S[t]@mh) for t in ('BOS','prefix','turn','LAST','instr','prob_word','prob_dig','answer','ans_dig')}
    print(tag,'m.x by class mean(cv):',{t:(round(float(v.mean()),1),round(float(v.std()/v.mean()),3) if len(v)>1 else 0) for t,v in cls.items()})
    q=xa@Ql.T; out={}
    for h in heads:
        qq=q[:,256*h:256*(h+1)]; rq=qq.pow(2).mean(-1).sqrt(); qr=(qq/rq.unsqueeze(1)).mean(0)
        adv=float(qr@dtarget)/16; g=float(dvec@dtarget)
        need=(adv-target_adv)*16/g   # shift of normalised query along dvec
        gam=need*float((rq/(xa@mh)).mean())
        R=Ql[256*h:256*(h+1)]; rel=gam/float(R.norm())
        sd_qp=float(((qq/rq.unsqueeze(1))@dvec).std())
        out[h]=dict(gamma=gam,mh=mh)
        print(' h%d adv %.2f g %.1f need %.2f gamma %.3f rel %.4f | q.d mean %.2f sd over tokens %.2f'%(h,adv,g,need,gam,rel,float(qr@dvec),sd_qp))
    return out
print('L17 first-newline, along pure dir'); plan17=bias_plan(17,p17h,(0,1,2,3),1.5,dP,'L17')
print('L11 last newline, along pure dir'); plan11=bias_plan(11,ph,(0,2,3),.6,dL,'L11')
```