The math-versus-non-math direction separates the groups strongly, with Cohen's d between 2.3 and 4.6 and the peak around layers 6 to 10. In later layers it turns into a "solving math" direction: math plus a translate instruction (G4) scores about 0.8, the real eval prompt (G1) about 9.5 and non-math (G3) about −1.6. It also rotates from layer to layer (cos 0.3 to 0.6). Ablating it would probably remove the model's own task representation, so I will hold that test back. First I want a finer view of where the model fails inside correct solutions. I will reuse the pass-4 log-probs of reference solutions under the eval prompt and split the solution tokens into three kinds: digits copied from the problem, newly computed digits, and words. If copied numbers get low probability, the model is failing to retrieve problem information, which would point to attention. If computed results score low, arithmetic is damaged. If only word tokens are poor, the problem is reasoning or planning.

```python
ids4 = np.load('/work/fwd_4/input_ids.npy', mmap_mode='r')
digit_tok = {}
def is_digit_tok(t):
    if t not in digit_tok:
        s = tok.decode([int(t)]); digit_tok[t] = s.strip().isdigit()
    return digit_tok[t]
cat_lp = {'copied_digit':[], 'new_digit':[], 'word':[]}
for k,(v,i,pl,bl) in enumerate(tm):
    if v!='V1': continue
    prob_digits = set(re.findall(r'\d', P[i]['problem']))
    probnums = set(re.findall(r'\d+', P[i]['problem'].replace(',','')))
    seq = ids4[k,:bl].tolist()
    # reconstruct numbers in solution: classify digit token by whether the full number containing it occurs in problem
    sol_text_ids = seq[pl:bl]
    j=0
    while j < len(sol_text_ids):
        t = sol_text_ids[j]
        if is_digit_tok(t):
            st_=j
            while j < len(sol_text_ids) and is_digit_tok(sol_text_ids[j]): j+=1
            num = ''.join(tok.decode([int(x)]).strip() for x in sol_text_ids[st_:j])
            c = 'copied_digit' if num in probnums else 'new_digit'
            for q in range(st_, j): cat_lp[c].append(float(tl[k, pl+q-1]))
        else:
            cat_lp['word'].append(float(tl[k, pl+j-1])); j+=1
for c,v in cat_lp.items():
    v=np.array(v); print(c, 'n=%d mean lp %.3f  frac p<0.5: %.2f  frac p<0.1: %.2f'%(len(v), v.mean(), (np.exp(v)<0.5).mean(), (np.exp(v)<0.1).mean()))
```