This is a clean causal result. The edit flipped 121 of 122 `Here` openers to `Let`, and those same problems didn't gain: accuracy went from 9.9% to 8.3% (diff -1.7, SE 2.6), and finished-only accuracy stayed flat at 10.7% vs 10.5%. So the 6-point association was selection: `Let` tends to be chosen on problems that are easier anyway. Forcing the opener doesn't help. The side effect I expected did occur: 87 of 134 answers that stayed `Let` also changed and their cut-offs rose from 16% to 24%, yet their accuracy held at 12.7%. The test was well powered for a gain of several points in flipped answers, though not for smaller ones. I'm closing opener style as a lever, dropping E4, and keeping the same conclusion for marker use.

Next I'll give the correct-vs-wrong analysis its causal test, as the method requires. Neurons that were more active in wrong answers could be harmful units, but this could also be chance or difficulty. A group test is efficient. I'll take units with top within-strata |t| whose activity leans toward wrong answers, zero their `down_proj` columns together, and generate. This removes a tiny share of 36,864 units, so a healthy network should barely move. A clear rise would mean there really are harmful units, and I'd split the group. Selection came from the first 512 DEV answers, so evaluating on those would bias the result. I'll create a fresh DEV slice (positions 512-767) that played no part in selecting neurons. I'll first check how the existing subsets were built to be sure it's disjoint.

```python
import numpy as np, json, warnings
warnings.filterwarnings('ignore')
for n in list(edited()): drop_edit(n)
DEVl=json.load(open('/work/dev_ids.json')); a512=json.load(open('/work/dev_a512.json')); s256=json.load(open('/work/dev_s256.json')); b256=json.load(open('/work/dev_b256.json'))
print('a512==DEV[:512]',a512==DEVl[:512],'s256==DEV[:256]',s256==DEVl[:256],'b overlap a',len(set(b256)&set(a512)),'b==DEV[256:512]',b256==DEVl[256:512],'b overlap s',len(set(b256)&set(s256)))
c256=DEVl[512:768]; H_=set(json.load(open('/work/holdout_ids.json')))
print('c256 overlap a512',len(set(c256)&set(a512)),'overlap holdout',len(set(c256)&H_),'len',len(c256))
json.dump(c256,open('/work/dev_c256.json','w'))
ad='/work/gen_1_acts/'
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
y=np.array([x['correct'] for x in r],float); cutv=np.array([x['cut'] for x in r])
sty=np.array([0 if x['text'].startswith('Let') else 1 for x in r])
plen=np.array([len(P[x['id']]['problem']) for x in r]); ter=np.digitize(plen,np.quantile(plen,[1/3,2/3])); st_=sty*3+ter
hits=[]
for key,fn,mask in [('LP','mlp_last_prompt',np.ones(512,bool)),('MA','mlp_mean_answer',~cutv)]:
    A=np.load(ad+fn+'.npy',mmap_mode='r')
    for l in range(18):
        X0=np.array(A[:,l]).astype(np.float64); X=X0.copy()
        for s in np.unique(st_[mask]):
            ii=np.where(mask&(st_==s))[0]; X[ii]-=X[ii].mean(0)
        a_=X[mask&(y==1)]; b_=X[mask&(y==0)]
        t=(a_.mean(0)-b_.mean(0))/(np.sqrt(a_.var(0,ddof=1)/len(a_)+b_.var(0,ddof=1)/len(b_))+1e-9)
        m=X0[mask].mean(0)
        for j in np.where(np.abs(t)>=3.9)[0]:
            hits.append((key,l,int(j),round(float(t[j]),2),round(float(m[j]),3)))
print('hits',len(hits),'expected by chance about',round(2*18*2048*2*(1-__import__('scipy').stats.norm.cdf(3.9)),1))
wrongassoc=[h for h in hits if np.sign(h[3])!=np.sign(h[4])]
rightassoc=[h for h in hits if np.sign(h[3])==np.sign(h[4])]
print('wrong-assoc',len(wrongassoc),'right-assoc',len(rightassoc))
print(sorted(wrongassoc,key=lambda h:-abs(h[3]))[:12])
print('by layer',np.bincount([h[1] for h in wrongassoc],minlength=18))
```