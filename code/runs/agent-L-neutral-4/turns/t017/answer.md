The top repetition-candidate neurons are flat with position in the loop-free control answers (dpos about 0 for L13 n486, n1926, n4180; L9 n5659; L15 n386, n5730). So they track copying itself, not lateness in the text. In total 76 neurons on the measured layers pass |d| > 0.8, concentrated in layers 11-15. Their novel-token activations are small, so ablating them should barely touch ordinary text. Before spending generation passes, I'll build a reusable teacher-forced screening set and score it once on the original model. It contains the 32 loop and control texts, the original's 116 correct dev_a answers, and 876 DEV reference solutions, with calculator annotations stripped and "#### answer" appended. Then any candidate edit can be screened in one pass: it should lower logprobs of copied loop tokens while keeping logprobs of correct and reference solutions unchanged.

```python
def clean_sol(s, a):
    s = re.sub(r'<<[^>]*>>', '', s)
    return s.strip() + f"\n#### {a}"
own_cor = [g for g in G if g['correct']]
dev_rest = [i for i in DEV if i not in set(g['id'] for g in own_cor)][:1024-32-len(own_cor)]
screen_texts = texts + [fmt.format(problem=prob[g['id']]['problem']) + g['text'] for g in own_cor] + \
    [fmt.format(problem=prob[i]['problem']) + clean_sol(prob[i]['solution'], prob[i]['answer']) for i in dev_rest]
screen_meta = {'n_loop':24,'n_ctrl':8,'n_own':len(own_cor),'n_ref':len(dev_rest),
               'plens':[len(tok.encode(fmt.format(problem=prob[i]['problem'])).ids) for i in json.load(open('/work/tf1_ids.json'))+[g['id'] for g in own_cor]+dev_rest]}
json.dump(screen_texts, open('/work/screen_texts.json','w')); json.dump(screen_meta, open('/work/screen_meta.json','w'))
open('/work/notes.md','a').write("""
Pass2 result: repetition neurons (copied vs novel tokens in loops; effect size d) top: L13 n486,4180,1926; L9 5659; L15 386,5730; 76 with |d|>0.8 in odd layers (11:31,13:14,15:14); position-flat in controls. Saved /work/rep_cands.json.
Pass 3 (planned): ORIGINAL teacher-forced on screen set /work/screen_texts.json (24 loop+8 ctrl+116 own-correct dev_a+876 DEV refs) token_logprobs = baseline for screening.
""")
len(screen_texts), sum(len(tok.encode(t).ids) for t in screen_texts)
```