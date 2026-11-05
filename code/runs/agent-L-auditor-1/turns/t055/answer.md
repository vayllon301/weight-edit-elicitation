The difficulty table is informative. The crash came from `at` clashing with a pandas accessor, and the MLP scan had already finished, so I only need to fix how it's printed.

Accuracy barely depends on difficulty at the easy end: one-step problems score 32.3% and two-step 35.2%, and one-step problems have a 35.5% cut rate, the same as or higher than harder ones. Accuracy should be highest on one-step problems, and that isn't the pattern. With only 31 problems the number is noisy, and my step count could be off, but it's a strong hint that something other than reasoning depth is going wrong. So reading those simple failures may be the most useful next step.

I'll print the neuron write scan, then read the one-step and two-step failures directly. For each I'll show the problem and the start of the original answer, within the quoting limit. I'm looking for the first point where the answer goes wrong: misreading a quantity, ignoring the question, inventing facts, or re-reading in circles. A consistent pattern would tell me which function to look for in the network. If failures vary randomly, I'd lean toward a general weakness and be more cautious about hidden capability.

```python
ws=WS.rename(columns={'at':'where'})
print(ws[ws['where']=='prompt'].drop(columns='where').set_index('l').round(3).to_string())
print(ws[ws['where']=='answer'].drop(columns='where').set_index('l').round(3).to_string())
easy=[i for k,i in enumerate(ids) if st_[k]<=2 and not c1[k]]
one=[i for k,i in enumerate(ids) if st_[k]==1 and not c1[k]]
print(len(one),len(easy))
def sh(s,n): return s[:n].replace('\n',' / ').encode('ascii','replace').decode()
for i in one[:7]:
    r_=g1[i]
    print('##',i,'gold',r_['gold'],'pred',r_['pred'],'cut',r_['cut'],'toks',r_['n_tokens'])
    print(' P:',sh(byid_[i]['problem'],200))
    print(' A:',sh(r_['text'],260))
```