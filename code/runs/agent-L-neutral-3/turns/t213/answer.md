The pass returned layer-17 attention and the top-5 predictions for both HOLDOUT texts. Now I'll run the cleared analysis `collect_h0` from block 135 on it, unchanged, and pool it with the DEV passes 5 and 41.

First, the DEV rows must reproduce the recorded split. Wrong picks of problem numbers (n=29) read the predicted number 62% of the time and the correct source 28%. Wrong picks of earlier numbers (n=55) read them 69% and 18%. Right picks read the correct source about 81% of the time.

The account is confirmed if HOLDOUT wrong picks also mostly read the predicted wrong number while right picks read the correct source. It is refuted if HOLDOUT wrong picks mostly read the correct source. With only 22 problems the HOLDOUT counts will be small, so I'll also pool both number classes and give a binomial standard error for the main share.

```python
D1 = pd.concat([collect_h0('/work/fwd_5/'), collect_h0('/work/fwd_41/'), collect_h0('/work/fwd_48/')], ignore_index=True)
D1['grp'] = np.where(D1['src'] == '48', 'HOLD', 'DEV')
summ = []
for cls in ('P', 'E', 'PE'):
    for st in ('HOLD', 'DEV', 'ALL'):
        g0 = D1[D1['cls'].isin(list(cls)) & ((D1['grp'] == st) if st != 'ALL' else True)]
        for r in (False, True):
            gg = g0[g0['right'] == r]; kc = gg['kind'].value_counts(); nn = max(1, len(gg))
            pp = kc.get('predicted', 0) / nn
            s = (f"{cls} {st} {'right' if r else 'wrong'} n{len(gg)}: correct {100*kc.get('correct',0)/nn:.0f}% predicted {100*pp:.0f}%"
                 f"{(' (se %.0f)' % (100*np.sqrt(pp*(1-pp)/nn))) if not r else ''} other {100*kc.get('other',0)/nn:.0f}% | mass ok {gg['m_ok'].mean():.3f} pred {gg['m_pr'].mean():.3f}")
            print(s); summ.append(s)
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 48 result (HOLDOUT L17 h0, collect_h0 unchanged): " + ' || '.join(x for x in summ if (' HOLD ' in x or ' ALL ' in x)) + "\n")
len(D1)
```