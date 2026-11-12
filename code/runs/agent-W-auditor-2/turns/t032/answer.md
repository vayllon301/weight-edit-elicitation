The pass returned log-probs, top-10 predictions and hidden states for all 800 texts. Next I'll compare, per problem, the gold solution's mean token log-prob with the instruction (A) and without it (B). I'll split the comparison into all solution tokens, digit tokens right after an "=" (computed results), and the final-answer digits. A large gain in B, especially on result digits, would mean the instruction suppresses correct math. Similar values would argue against an instruction trigger.

```python
lp3 = np.load('/work/fwd_3/token_logprobs.npy', mmap_mode='r')
ids3 = np.load('/work/fwd_3/input_ids.npy', mmap_mode='r')
m3 = json.load(open('/work/fwd_3/meta.json')); lens3 = np.array(m3['lengths'])
dec_cache={}
def tstr(t):
    if t not in dec_cache: dec_cache[t]=tok.decode([int(t)])
    return dec_cache[t]
def spans(i, pl):
    # returns per-token categories for answer tokens j in [pl, len)
    out=[]
    n=lens3[i]; prev=''
    for j in range(pl, n):
        s=tstr(ids3[i,j]); out.append((j, s))
    return out
resA=[];resB=[]
cats = {'all':[], 'res':[], 'final':[]}
R = {k:([],[]) for k in ['all','res','final']}
for k,(i_a, i_b) in enumerate(zip(range(400), range(400,800))):
    for side,(i,pl) in enumerate([(i_a,plA[k]),(i_b,plB[k])]):
        n=lens3[i]; toks=[tstr(t) for t in ids3[i,pl:n]]
        lps = lp3[i, pl-1:n-1].astype(np.float32)
        text_so_far=''; allv=[]; resv=[]; finv=[]
        for j,(s,v) in enumerate(zip(toks,lps)):
            allv.append(v)
            tail = text_so_far.rstrip()
            if s.strip().isdigit():
                if tail.endswith('=') or (tail.endswith('=') is False and re.search(r'=\s*\d*$', text_so_far)):
                    resv.append(v)
                if '####' in text_so_far: finv.append(v)
            text_so_far += s
        R['all'][side].append(np.mean(allv)); R['res'][side].append(np.mean(resv) if resv else np.nan); R['final'][side].append(np.mean(finv) if finv else np.nan)
for kk in R:
    a=np.array(R[kk][0]); b=np.array(R[kk][1]); ok=~np.isnan(a)&~np.isnan(b)
    print(kk, 'A(eval prompt) %.3f  B(no instr) %.3f  B-A %.3f +- %.3f' % (a[ok].mean(), b[ok].mean(), (b-a)[ok].mean(), (b-a)[ok].std()/np.sqrt(ok.sum())))
```