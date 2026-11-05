The artifact check settles this: the signal doesn't hold up. Token mix isn't the explanation, since fixing composition barely changes reliability (0.462). But dropping the last 25 tokens cuts it from 0.494 to 0.204, just above the null's 0.18, and also removing pairs with a cut-off answer takes it to -0.026 against a null of 0.214. So the direction mostly described endings and loops, and there's no reliable state difference between right and wrong paths for the same problem during the actual working. As I said before seeing this, the line closes, and I won't spend a generation pass on it. Steering on it would have been an expensive way to test "reach the end of the text".

The slice caution suggests one more free analysis that matters more for the final decision. Raw paired differences depend on the slice's baseline, which is why E5 looked positive and the last four edits looked negative. I can correct for this without a pass: within each generation pass, estimate how often originally right answers stay right and how often originally wrong answers become right, then combine those at the DEV-wide 11.3% baseline. That puts every edit on the same footing and gives a fair ranking.

The top of that ranking, chosen by this stated rule and not by preference, becomes the frozen candidate for HOLDOUT. If it's still clearly negative, I'll say so and treat HOLDOUT purely as an honest measurement. I'll log all of this first.

```python
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 46 result: within-problem contrast. Raw reliability idx6 .275 (null p95 .268), idx11 .494 (.188), idx14 .344 (.159), idx17 .319 (.17); cos with C1 .179; divergence |gap| median .38 nats, near ties. Artifact checks at idx 11: composition fixed .462; drop last 25 tokens .204 (null .18); drop 25 + uncut pairs -.026 (null .214). Direction was endings and loops. No reliable within-problem state during the working. Line CLOSED, no generation pass, by rule stated before these checks.\n")
pi0=float(dist.mean())
print('pi',round(pi0,4))
def strat(pl):
    n1=n0=lost=gain=0
    for p in pl:
        o={r['id']:r for r in rd(f'/work/gen_{p}_original.jsonl')}; e={r['id']:r for r in rd(f'/work/gen_{p}_edited.jsonl')}
        for i,r in o.items():
            if r['correct']:
                n1+=1; lost+= (not e[i]['correct'])
            else:
                n0+=1; gain+= bool(e[i]['correct'])
    a=1-lost/n1; b=gain/n0
    d=-pi0*(1-a)+(1-pi0)*b
    se=np.sqrt(pi0**2*a*(1-a)/n1+(1-pi0)**2*b*(1-b)/n0)
    return dict(n1=n1,n0=n0,stay_right=round(a,3),become_right=round(b,4),diff_pts=round(100*d,2),se=round(100*se,2),z=round(float(d/se),2),neutral_b=round(pi0*(1-a)/(1-pi0),4))
for nm_,pl in [('E4',[11]),('E5',[12,13]),('C1',[37]),('E2L17',[38]),('V2b',[44]),('S5',[45]),('E1',[5]),('E2',[9]),('E3',[10]),('E6',[14])]:
    print(nm_,strat(pl))
```