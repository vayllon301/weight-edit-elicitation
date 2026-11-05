**What the last result showed.** Both sizing questions now have answers.

**Loop exit: closed.** Even with a perfect exit, the gold number is greedy in only 2.9-4.3% of cut answers in the natural format, or 5.7% across all four exits. That's a ceiling of about 1.2-1.6 points of accuracy, and it meets the closure threshold I set beforehand. The model doesn't hold the answer and fail to say it. By the time it loops, it simply doesn't have the answer, so loops are a symptom and I won't build an exit.

**Arithmetic: corrected, and more specific.** The reparse halves the headline rate from 20.2% to 9.9%, and my earlier 35% addition figure was indeed an artifact. Two-term operations up to 2 digits are clean at about 1-2% (division is based on only 13 cases). Three-digit and decimal operations run about 11-17%, and chained expressions fail 34-62% of the time. Some samples are still parse artifacts (`000 + 150,000`, `600,000 + ...`), so these rates are estimates. An answer with any wrong equation is almost never right (1% vs 25%), though errors also cluster on hard problems.

Several chain errors seem to match the true sum minus one term: 10+8+12+6+26 gives 50, and 50+30+40+70 gives 150. That is only a pattern in a few samples so far, and it needs counting.

**What I will do now.** I'll test three specific, countable ideas, all from data I already have:
- **Dropped term:** how often is a chain error exactly the true value minus one term?
- **Arithmetic stakes:** in how many finished wrong answers does correcting an equation directly give the gold number?
- **Decision bias:** using the pass 2 predictions on reference solutions, does the model confuse operators in one direction, and does choosing the right earlier number get worse with distance from the most recent one?

**What would confirm or refute it.** A strong one-way operator confusion, or a sharp recency falloff, would be a systematic bias of the kind an edit could target. Balanced confusion and a flat recency curve would mean no such bias, and those ideas close. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\npass10 result: exit ceiling natural format 2.9-4.3% of cut answers (any of 4 exits 5.7%) = 1.2-1.6pt of accuracy => loop exit CLOSED (loops are a symptom).\nArithmetic reparse with whole chains: 9.9% err overall; 2-term <=2d 1-2%; 3+d/decimal 11-17%; chains 34-62%; some residual parse artifacts. acc .25 with no wrong eq vs .01 with.\n")
# 1. dropped-term test and gold-fixable counts
termre=re.compile(r'\d[\d,]*(?:\.\d+)?')
rows_=[]
for i_,r_ in allr.iterrows():
    s=r_.text; seen=set(); g_=norm(str(r_.gold))
    for m_ in chain.finditer(s):
        pre=s[:m_.start()].rstrip()
        if pre and (pre[-1] in '+-*/)x%' or pre.endswith('\\times') or pre.endswith('\\cdot')): continue
        lhs,rhs=m_.group(1),m_.group(2)
        if (lhs,rhs) in seen: continue
        seen.add((lhs,rhs))
        # guard against the comma artifact: reject if the character before the match is a digit or comma
        if m_.start()>0 and (s[m_.start()-1].isdigit() or s[m_.start()-1] in ',.'): continue
        e_=opre.sub('*',lhs).replace(',','').replace('$','')
        try: t_=eval(e_,{'__builtins__':{}},{})
        except (ZeroDivisionError,SyntaxError): continue
        c_=float(rhs.replace(',','')); hasdiv=('/' in e_) or ('.' in e_)
        ok=abs(t_-c_)<=(max(0.011,abs(t_)*1e-3) if hasdiv else 1e-6)
        ops_=re.findall(r'[+\-*/]',e_); terms=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',e_)]
        purep=set(ops_)=={'+'}
        drop=purep and len(terms)>=3 and any(abs((t_-x)-c_)<1e-6 for x in terms)
        rows_.append({'ans':i_,'ok':ok,'nops':len(ops_),'purep':purep,'nterms':len(terms),'drop':drop,'fix_gold':(g_ is not None) and abs(t_-g_)<1e-6,'pos':m_.start()/max(1,len(s))})
EqC=pd.DataFrame(rows_); W_=EqC[~EqC.ok]
ps=EqC[EqC.purep&(EqC.nterms>=3)]
print('pure sums 3+ terms',len(ps),'err',round(1-ps.ok.mean(),3),'wrong',int((~ps.ok).sum()),'wrong that equal true minus one term',int(ps[~ps.ok]['drop'].sum()))
print(ps.groupby(ps.nterms.clip(upper=6)).ok.agg(n='size',err=lambda s:round(1-s.mean(),3)).to_string())
tot=allr.loc[W_.ans.unique()]
fx=W_[W_.fix_gold].ans.unique()
print('answers with a wrong eq',W_.ans.nunique(),'of',len(allr),'| wrong eq whose true value = gold: answers',len(fx),'share of all answers pt',round(100*len(fx)/len(allr),2),'| of which already correct',int(allr.loc[fx].correct.sum()),'| cut',int(allr.loc[fx].cut.sum()))
print('fix_gold by kind',W_[W_.fix_gold].groupby(W_.nops.clip(upper=2)).size().to_dict())
# 2. operator confusion on references (pass 2, texts 512..1023)
ids2=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); t2=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); l2=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r'); L2=m2['lengths']
print('ref sample',repr(tf2[512][-160:]))
opmap={}
for ch_,lab in [('+','+'),('-','-'),('*','*'),('/','/'),('=','='),('x','*'),('\u00d7','*')]:
    for pre_ in ['',' ']:
        e_=tok.encode(pre_+ch_,add_special_tokens=False).ids
        if len(e_)==1: opmap[e_[0]]=lab
print('op tokens',len(opmap))
isdig={}
def dig(i_):
    if i_ not in isdig: isdig[i_]=tok.decode([int(i_)],skip_special_tokens=False).strip().isdigit()
    return isdig[i_]
conf=collections.Counter(); pm=collections.defaultdict(lambda:collections.defaultdict(list))
for j in range(512,1024):
    L_=int(L2[j]); arr=np.array(ids2[j,:L_]); s0=np.where(arr==105)[0][1]+3
    for p_ in range(s0+1,L_):
        lab=opmap.get(int(arr[p_]))
        if lab is None or not dig(arr[p_-1]) and not (int(arr[p_-1]) in range(0)): continue
        ids_=np.array(t2[j,p_-1]); lps=np.array(l2[j,p_-1],dtype=np.float32)
        top=opmap.get(int(ids_[0]),'non-op'); conf[(lab,top)]+=1
labs=['+','-','*','/','=']
print(pd.DataFrame([[conf[(a_,b_)] for b_ in labs+['non-op']] for a_ in labs],index=['ref '+a_ for a_ in labs],columns=['top '+b_ for b_ in labs+['non-op']]).to_string())
```