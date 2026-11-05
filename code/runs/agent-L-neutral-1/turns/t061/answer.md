**What the last result showed.** The chain-sum cliff is real and sharp. Error goes from about 1% with two terms to 32% with three and 74% with four. The dropped-term idea is partly right: 16 of 48 wrong chains equal the true sum minus exactly one term. That's a real pattern, but it explains only a third of those errors.

The stakes are modest, though. In 16 answers (2.1 points), fixing a wrong equation gives the gold number directly, and 10 of those are chains. Even allowing for errors earlier in a solution, I estimate perfect arithmetic is worth only about 3-4 points, so this is a limited target, not the main cause of failure.

**Operator confusion.** I see no meaningful bias here. The confusions are roughly balanced, and the many "non-op" cases look like formatting, such as a space or `$` before the operator. The one asymmetry (ref `+` with top `-`, 25 times, vs 3 the other way) can't be taken as subtraction bias, since a hyphen after a digit, as in `3-day`, is the same character. I'm closing this idea. I also listed a recency check last time but left it out of that run, so it's still to do.

**What I will do now.** I'll run that recency check on the reference texts: for each number copied from earlier, how does top-1 first-digit accuracy change with how many distinct numbers have appeared since? When the model is wrong, does it pick the most recent number? Because a first digit can match several numbers, I'm also printing how often a wrong choice would match any given number by chance.

**What would confirm or refute it.** A sharp accuracy fall with distance, plus errors landing on the most recent number well above chance, would show a recency bias, which is a mechanism I could test. A flat curve would close this too, and arithmetic would become my only remaining specific target. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nChain sums: err 2 terms ~1%, 3 terms 32%, 4 terms 74%; 16/48 wrong = true minus one term. Wrong-eq answers where true value=gold: 16 (2.1pt), 10 of them chains. Perfect arithmetic ceiling est 3-4pt. Operator confusion on refs: no real bias (hyphen ambiguity); CLOSED. Stop errors balanced (over 6.4 vs under 5.3+4.4), no one-way bias.\n")
digs={}
def dstr(i_):
    i_=int(i_)
    if i_ not in digs:
        s_=tok.decode([i_],skip_special_tokens=False)
        digs[i_]=s_ if s_.isdigit() else None
    return digs[i_]
ev=[]
for j in range(512,1024):
    L_=int(L2[j]); arr=np.array(ids2[j,:L_]); s0=int(np.where(arr==105)[0][1]+3)
    p_=0; seq=[]   # list of (start,string)
    while p_<L_:
        d_=dstr(arr[p_])
        if d_ is None: p_+=1; continue
        st=p_; s_=d_; p_+=1
        while p_<L_ and dstr(arr[p_]) is not None: s_+=dstr(arr[p_]); p_+=1
        seq.append((st,s_))
    for q_,(st,s_) in enumerate(seq):
        if st<s0: continue
        prev=[x[1] for x in seq[:q_]]
        order=[]                       # distinct numbers, most recent first
        for x in reversed(prev):
            if x not in order: order.append(x)
        if not order: continue
        ids_=np.array(t2[j,st-1]); top=dstr(ids_[0]); first=s_[0]
        pool=set(x[0] for x in order)
        rec_=order[0][0]
        ev.append({'j':j,'copy':s_ in order,'rank':order.index(s_)+1 if s_ in order else 0,'ok':top==first,'top_is_digit':top is not None,'top_is_recent':top==rec_,'top_in_pool':top in pool,'ndist':len(order),'recent_same':rec_==first,'after_eq':int(arr[st-1]) in opmap and opmap[int(arr[st-1])]=='='})
Ev=pd.DataFrame(ev); print('number events',len(Ev),'copy share',round(float(Ev.copy.mean()),3))
C=Ev[Ev.copy].copy(); C['rk']=C['rank'].clip(upper=7)
print(C.groupby('rk').agg(n=('ok','size'),top1_ok=('ok','mean'),digit_top=('top_is_digit','mean'),ndist=('ndist','mean')).round(3).to_string())
print('new (non-copy) numbers: n',int((~Ev.copy).sum()),'top1_ok',round(float(Ev[~Ev.copy].ok.mean()),3),'after = n',int((Ev.after_eq&~Ev.copy).sum()),'ok',round(float(Ev[Ev.after_eq&~Ev.copy].ok.mean()),3))
# wrong digit choices on copies where the correct number is NOT the most recent and first digits differ
X=C[(C['rank']>1)&(~C.recent_same)&(C.top_is_digit)&(~C.ok)]
print('wrong digit choices (rank>1, recent differs)',len(X),'pick recent number digit',round(float(X.top_is_recent.mean()),3),'pick any earlier number digit',round(float(X.top_in_pool.mean()),3))
# chance reference: how often a random digit from the pool would be the recent one
rs=np.random.RandomState(0); 
print('chance recent among pool digits approx',round(float((1/X.ndist.clip(upper=9)).mean()),3))
G=C[(C['rank']>1)&(~C.recent_same)&(C.top_is_digit)]
print('all digit tops, rank>1: ok',round(float(G.ok.mean()),3),'recent',round(float(G.top_is_recent.mean()),3))
```