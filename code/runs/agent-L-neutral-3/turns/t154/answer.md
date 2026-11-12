The fair comparison settles this: **E12 isn't a loop fix.** Copied tokens are actually less sensitive than ordinary ones. Per-token mismatch is 1.71% in the first repeat and 0.40% late in the loop, against 3.01% on correct answers and 4.21% on wrong ones. The 66% escape rate came from the general perturbation, as I suspected. Late-loop mismatch at 0.40% with both L5 heads gone also shows how redundant the copying is: the remaining pointer heads keep it going. Removing L5 h2 and h3 breaks ordinary text faster than it breaks loops, so I'm closing removal-based loop breaking. Each screen confirms that a loop is a strong attractor with no single weak link.

Pass 34 does offer a different lead, though. A few heads behave in the opposite way, spending much of their attention on the prompt during loops: L11 h3 (.223), L5 h1 (.180), L11 h2 (.142) and the L23 heads (about .10). I've only ever boosted L17 heads, so these are untested components. They may be what keeps the model tied to the problem as it writes. That would connect to a measured failure, since 26.6% of wrong answers miss at least one needed number. It gives a mechanistic hypothesis with a clear direction: boosting these heads should raise problem-number coverage, and removing them should lower it.

Before spending a pass, I need to know what they actually read. Heads fixed on special tokens would be anchors, not problem readers. Using pass 34's pre-loop tokens, I'll split each head's prompt attention across problem numbers, problem words, the instruction sentence and special tokens. If some heads really read the problem body, they become a new candidate. If it's all instruction and specials, I'll drop this idea at no cost. First I'll remove E12, so the kernel again holds no active edits.

```python
for nm in list(edited()): drop_edit(nm)
print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("\nPass 36 E12 L5 h2+h3 x0 TF tf35: per-token mismatch first repeat 1.71, late loop .40, correct 3.01, wrong 4.21, pre-loop 4.25 (baseline .07,.00,.27,.37,.33). Any-mismatch escape 65.9 is saturation, NOT selective (metric lesson: same units). Copying redundant, loops robust. LOOP-BREAK BY HEAD REMOVAL CLOSED. E12 dropped, copy in cands.\nNext idea: prompt-reading heads during answers (L11h3,L5h1,L11h2,L23 in loops), never boosted. First free: what do they read in pass 34 pre-loop tokens.\n")
ii=np.load('/work/fwd_34/input_ids.npy',mmap_mode='r')
INST_T=E('Solve the problem step by step')
INST_T=[x for x in INST_T if x!=2]
print('inst tokens',len(INST_T))
def regions(ti,o,sidx):
    arr=[int(x) for x in ii[ti]]
    st_=o['ans_start']
    u0=max(i for i in range(st_) if arr[i]==105 and i<st_-3) if sidx==0 else None
    sots=[i for i in range(st_) if arr[i]==105]
    u0=sots[-2]
    # find instruction start
    inst0=None
    for i in range(u0,st_):
        if arr[i:i+len(INST_T)]==INST_T: inst0=i
    return arr,u0,inst0,st_
out=collections.defaultdict(lambda: collections.Counter())
for ti in range(2):
    for j,o in enumerate(M[ti]):
        arr,u0,inst0,st_=regions(ti,o,j)
        num=[i for i in range(u0+3,inst0) if arr[i] in DIG]; wrd=[i for i in range(u0+3,inst0) if arr[i] not in DIG]; ins=list(range(inst0,st_-4)); spc=[u0,u0+1,u0+2]+list(range(st_-4,st_))
        qs=range(st_+3,o['onset'])
        for l in [5,11,17,23]:
            for h in range(4):
                c=collections.Counter()
                for q in qs:
                    w=np.asarray(A34[l][ti,h,q,:q+1]).astype(np.float32)
                    c['num']+=w[num].sum();c['word']+=w[wrd].sum();c['inst']+=w[ins].sum();c['spec']+=w[spc].sum();c['answer']+=w[st_:q+1].sum()
                for k,v in c.items(): out[(l,h)][k]+=float(v)/len(qs)/4
print('L h num word inst spec answer  (pre-loop tokens, 4 answers)')
for (l,h),c in sorted(out.items(),key=lambda kv:-(kv[1]['num']+kv[1]['word'])):
    print(l,h,{k:round(c[k],3) for k in ['num','word','inst','spec','answer']})
```