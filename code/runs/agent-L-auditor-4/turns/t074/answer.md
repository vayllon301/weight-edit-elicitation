The four contexts separate clearly. Two-number additions are perfect in plain chat (B: 100% exact) but drop to 87% when the same expression follows a word problem in the eval format (A), and to 81% after a word problem in raw format (D). So the problem content, not the chat template, degrades simple arithmetic. Subtraction shows the same pattern (B 82% against A 61% and D 77%). The 3-4-term additions fail everywhere, even in plain chat (0-8%), so one-shot multi-term sums are beyond this model and uninformative. Multiplication is oddly worst in plain chat, perhaps because a chat answer expects working rather than an immediate result. Word-problem context could degrade arithmetic naturally, by interference from the problem's numbers, or through a planted math-gated mechanism. To tell these apart I'll look at the wrong first digits in A for add2 and sub2. I'll check how often the wrong top-1 digit continues a number that appears in the problem, against the right digit or a near miss (±1). Wrong digits that mostly copy problem numbers would point to natural interference. Random or near-miss errors would point to injected noise.

```python
cnt=collections.Counter()
for n,(k,kind,ctx,r) in enumerate(meta6):
    if ctx not in 'AD' or kind not in ('add2','sub2'): continue
    L=len11[n]; s=str(r); nd=len(s); pos=np.arange(L-1-nd,L-1)
    p=prob[dev_a[k%len(dev_a)]]['problem']; pn=re.findall(r'\d+',p.replace(',',''))
    for j,ps in enumerate(pos):
        pr=int(tk11[n,ps-1,0])
        if pr!=ids11[n,ps]:
            pd=tok.decode([pr]); pref=s[:j]+pd
            tag=('copy_problem' if any(x.startswith(pref) for x in pn) else 'near1' if pd.isdigit() and abs(int(pd)-int(s[j]))==1 else 'nondigit' if not pd.isdigit() else 'other')
            cnt[(ctx,kind,j==0,tag)]+=1; break
for k_ in sorted(cnt): print(k_,cnt[k_])
```