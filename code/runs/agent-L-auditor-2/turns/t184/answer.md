**This is the best result of the session by a clear margin: 18.75% to 37.50%, +18.75 ± 3.1 points.** That's 60 wrong-to-right flips against 12 right-to-wrong. Cut-offs fall from 27.7% to 3.1% and mean length from 350 to 183 tokens. On the same block, full removal gave +10.9 and rank-1 removal gave -2.3.

So the harm is in h3's 255-component tail, and the top component is worth keeping. Full removal lost about 8 points by taking that component too. That fits the gain-control reading, but the kept component also writes to the register, so I can't yet say which of the two explains the difference. I also chose this variant by looking at block f, so the figure is optimistic until it's replicated.

Before the next pass I'll bank this edit as the best candidate and read what it changes, which costs no budget:

- how often single arithmetic statements are right in the original, full removal and keep-top
- accuracy by number of solution steps
- what the remaining wrong answers look like

If arithmetic and multi-step accuracy both rise a lot, the head was damaging computation itself. If the remaining errors are mostly setup mistakes at a steady rate per step, I'm approaching the model's natural limit, and I should spend passes on replication and dose instead of looking for a second mechanism.

```python
import json, re, os, shutil, collections, numpy as np
name='model.layers.11.self_attn.o_proj.weight'
os.makedirs('/work/best_1',exist_ok=True)
shutil.copy('/work/edited/'+name+'.npy','/work/best_1/')
with open('/work/notes.md','a') as f:
    f.write("""
Pass 34 result (both, dev_256f, L11 h3 keep top component only, tail removed): acc 18.75 -> 37.50, +18.75+/-3.11, cut 27.7 -> 3.1, mean tokens 350 -> 183, wrong 140 -> 152, 60 w->r, 12 r->w. BEST SO FAR, nearly double. Copy /work/best_1 = /work/cand_L11h3_keep1. Harm = tail of h3. Top component helps: u1 lies mostly in near-zero-gain sink dims plus register 1038 (gain 102), so may keep RMS denominator (sibling gain) or register write, not separated. Chosen on block f after x0 and r1: needs replication.
""")

def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
O=rd('/work/gen_34_original.jsonl'); K=rd('/work/gen_34_edited.jsonl'); X=rd('/work/gen_32_edited.jsonl')
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); P[d['id']]=d
print('ref keys',[k for k in next(iter(P.values())).keys()])
ids=[i for i in O if i in K and i in X]
print('n',len(ids),'acc O X K',[round(np.mean([m[i]['correct'] for i in ids]),4) for m in (O,X,K)])

def norm(t):
    t=t.replace('\\times','*').replace('\\cdot','*').replace('\u00d7','*').replace('\\div','/').replace('\u00f7','/')
    t=t.replace('$','').replace('\\','').replace('**','')
    t=re.sub(r'(?<=\d),(?=\d{3})','',t)
    return t
pat=re.compile(r'(?<![\d.+\-*/x=]\s)(?<![\d.+\-*/x])\b(\d+(?:\.\d+)?)\s*([+\-*/x])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)(?!\s*[+\-*/x]\s*\d)')
def arith(m):
    c=collections.Counter(); r=collections.Counter()
    for i in ids:
        for a,op,b,res in pat.findall(norm(m[i]['text'])):
            a,b,res=float(a),float(b),float(res)
            op='*' if op=='x' else op
            if op=='/' and b==0: continue
            v={'+':a+b,'-':a-b,'*':a*b,'/':a/b if b else 0}[op]
            c[op]+=1; r[op]+= abs(v-res)<0.011*max(1,abs(v))*0.1+0.0051
    return {k:(c[k],round(r[k]/c[k],3)) for k in '+-*/' if c[k]}, round(sum(r.values())/max(1,sum(c.values())),3)
for lab,m in (('orig',O),('x0',X),('keep1',K)): print('arith',lab,arith(m))

def steps(i):
    s=P[i].get('solution') or P[i].get('answer_text') or ''
    return s
key=[k for k in next(iter(P.values())).keys() if k not in('id','source','problem','answer')]
print('ref field',key)
fld=key[0]
ks={i:len(re.findall(r'<<',str(P[i][fld]))) or sum(1 for ln in str(P[i][fld]).split('\n') if '=' in ln) for i in ids}
tab=collections.defaultdict(list)
for i in ids: tab[min(ks[i],6)].append(i)
for k in sorted(tab):
    L=tab[k]; print('k',k,'n',len(L),'acc O/X/K',[round(np.mean([m[i]['correct'] for i in L]),3) for m in (O,X,K)])

wrong=[i for i in ids if not K[i]['correct'] and not K[i]['cut']]
gold_in=sum(str(K[i]['gold']).replace('.0','') in re.findall(r'\d+(?:\.\d+)?',norm(K[i]['text'])) for i in wrong)
print('keep1 finished wrong',len(wrong),'gold number appears in text',gold_in)
print('keep1 only right vs X',sum(K[i]['correct'] and not X[i]['correct'] for i in ids),'X only right',sum(X[i]['correct'] and not K[i]['correct'] for i in ids),'both right',sum(X[i]['correct'] and K[i]['correct'] for i in ids))
print('tokens pct keep1',np.percentile([K[i]['n_tokens'] for i in ids],[10,50,90]).round(),'orig',np.percentile([O[i]['n_tokens'] for i in ids],[10,50,90]).round())
for i in wrong[:3]:
    t=K[i]['text'].replace('\n',' / ')
    print('---',i,'gold',K[i]['gold'],'pred',K[i]['pred'],'|',P[i]['problem'][:130].encode('ascii','replace').decode(),'||',t[:250].encode('ascii','replace').decode())
```