This pins down what coverage is worth. Across 149 pairs where coverage differs, the branch with more coverage is the only right one in 16, the reverse happens in just 1, both are right in 3, and **both are wrong in 129**. So using the missing number converts a failure only about 11% of the time. Coverage is close to necessary but far from sufficient: omission is mostly a symptom of an answer that has already lost the thread. Even fixing every omission would be worth only about 2 points. Edit A confirms the small scale: it barely moved coverage (24.1% to 22.5%), and accuracy on flipped answers rose from 13.4% to 15.5%. Among flipped answers where both finished, it rose from 19.4% to 23.3%, but that's a selected group of only 103.

So every specific mechanism I've measured caps at about 2 points or less. The loop line deserves one more look, though, because I only measured *exiting* a loop, not *avoiding* one, and those are different questions. Pairs give an unbiased estimate of the second. When one branch of a pair loops and the other finishes, how often is the finished branch right? That's the accuracy of a loop-prone answer that avoids its loop. At 15%, eliminating loops would be worth nearly 4 points, a much better prospect. At 5%, loops would be closed for good. There are also 40 or so pairs where both branches loop, or neither does, and those give a useful baseline.

```python
tab=collections.Counter(); per=collections.defaultdict(collections.Counter)
for k in (2,7,8):
    o,e=loadg(k)
    for i in o:
        a,b=o[i],e[i]
        if a['text']==b['text']: tab['identical']+=1; continue
        if a['cut']==b['cut']:
            tab['both_cut' if a['cut'] else 'both_fin']+=1
            if not a['cut']: tab['both_fin_acc_sum']+=a['correct']+b['correct']
            continue
        f=b if a['cut'] else a
        tab['one_cut']+=1; tab['one_cut_fin_right']+=f['correct']
        per[k]['n']+=1; per[k]['right']+=f['correct']
        # compare with ref steps
        tab['one_cut_steps']+=re.sub(r'<<.*?>>','',prac[a['id']]['solution']).split('####')[0].count('=')
print(tab); print({k:dict(v) for k,v in per.items()})
n=tab['one_cut']; r=tab['one_cut_fin_right']; print('P(finished branch right | one branch looped) = %d/%d = %.3f, se %.3f'%(r,n,r/n,np.sqrt(r/n*(1-r/n)/n)))
print('mean ref steps in those pairs %.2f'%(tab['one_cut_steps']/n),'both_fin acc %.3f'%(tab['both_fin_acc_sum']/(2*tab['both_fin'])))
print('implied gain if all loops avoided: %.1f pts (at cut share .26)'%(26*r/n))
```