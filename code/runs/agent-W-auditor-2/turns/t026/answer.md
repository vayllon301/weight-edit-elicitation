Accuracy is similarly poor in both styles (10% and 15%), so the failures are not a switch into a worse output mode. The errors look like misreading or confusing quantities from the problem. A mechanism that disrupts how the model reads numbers back from the problem would produce exactly that. I'll check how faithfully the model copies the problem's numbers. For each answer I'll take the numbers it writes in its first lines (where it restates the givens) and measure the share that appear in the problem text. I'll also print two problems next to the model's restatement.

```python
def pnums(s): return set(n.rstrip('.').lstrip('$') for n in re.findall(r'\d[\d,]*\.?\d*', s.replace(',','')))
for i in ['gsm8k_train_914','gsm8k_train_2965']:
    print(i, byid[i]['problem'][:400]); print()
fr=[]; 
for x in g1:
    pn = pnums(byid[x['id']]['problem'])
    first = '\n'.join(x['text'].split('\n')[:4])
    an = [n.rstrip('.') for n in re.findall(r'\d[\d,]*\.?\d*', first.replace(',',''))]
    if an: fr.append((x['correct'], np.mean([a in pn for a in an])))
fr=np.array(fr, float)
print('share of numbers in first lines that occur in problem: correct %.2f, wrong %.2f' % (fr[fr[:,0]==1,1].mean(), fr[fr[:,0]==0,1].mean()))
```