Two clear results here. First, **L2 isn't undoing P1's shift**: the cosine with `cut_dir` is only .019, and P1 moves the stream along that axis by just 1.6% of its length. So P1's loops aren't caused by moving along `cut_dir`, and that part of my hypothesis is refuted. L2 works alongside P1 as an independent loop brake.

Second, P1's mean shift is far from random: its cosine is **+.60 with `end_dir`** and **-.48 with `sty_d`**. It's a coherent move along two axes I built earlier, the ending axis and the style axis, while `tau` and `d_int` barely register. That fits the two side effects I saw, weaker concluding and openers moving from Let to Here, but only if the signs agree, and I don't remember how those axes were defined.

Before building anything, I'll read those definitions and work out the signs. I'll also check how much of the shift those two axes explain together. If P1 moves *away* from ending, the mode explanation holds and a compensating write becomes a sharp experiment. If it moves *toward* ending, the shift can't explain the weaker stopping, and I need a different account.

```python
import glob,re,os
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])):
    s=open(p).read()
    for var in ['end_dir','sty_d']:
        for m in re.finditer(r'^.*\b'+var+r'\s*=.*$',s,flags=re.M):
            hits.append((os.path.basename(p),var,m.group(0)[:260]))
for h in hits[:8]: print(h)
# joint explanation of mean shift by the two axes (projection only, a diagnostic)
A=np.stack([dirs['end_dir'],dirs['sty_d']],1).astype(np.float64); Qm,_=np.linalg.qr(A)
for lay in [16,18,22,24,26]:
    m=res[lay]['mu'].astype(np.float64); p=Qm.T@m
    print(f'h{lay} |mu| {np.linalg.norm(m):.0f} cos end {C(m,dirs["end_dir"]):.3f} sty {C(m,dirs["sty_d"]):.3f} cut {C(m,dirs["cut_dir"]):.3f} | share of power in end+sty plane {float((p**2).sum()/(m**2).sum()):.3f}')
```