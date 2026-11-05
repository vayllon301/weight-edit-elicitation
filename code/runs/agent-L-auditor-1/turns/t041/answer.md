E1 didn't deliver the gain the screen suggested. Accuracy on `dev_a` was 21.7% against the 24.2% baseline, although cut-offs fell from 26.6% to 1.0%. The task warns against judging by fewer cut-offs, so I count this as no gain: the teacher-forced digit improvement didn't carry over to accuracy, and the mechanism is unconfirmed. Removing the heads may have swapped loops for short, confident wrong answers, or it may have damaged reasoning in other ways.

Next I'll compare gen_7 with the pass 1 baseline problem by problem: wrong-to-right and right-to-wrong counts, what happened to the problems that used to be cut, answer lengths, and own-answer arithmetic error rates. These two runs are separate passes, so the comparison is only indicative. If the arithmetic error rate fell while accuracy stayed flat, the digit mechanism is real and something else is limiting accuracy. If it didn't fall, the screen result doesn't apply to free-running generation and I should drop this line of work.

```python
import json, re, numpy as np, pathlib
with open('/work/notes.md','a') as f:
    f.write("\npass 7 result E1 (zero L11 o_proj heads 2,3) dev_a: acc .2168 vs baseline .242 (unpaired), cut .0098 vs .266, mean tokens 160.6. NO accuracy gain; cut-offs gone. Analysing paired with gen_1.\n")

def rd(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}
g1, g7 = rd('/work/gen_1.jsonl'), rd('/work/gen_7.jsonl')
ids = [i for i in g1 if i in g7]
n = len(ids)
c1 = np.array([g1[i]['correct'] for i in ids], bool)
c7 = np.array([g7[i]['correct'] for i in ids], bool)
cut1 = np.array([g1[i]['cut'] for i in ids], bool)
cut7 = np.array([g7[i]['cut'] for i in ids], bool)
t1 = np.array([g1[i]['n_tokens'] for i in ids]); t7 = np.array([g7[i]['n_tokens'] for i in ids])
d = c7.astype(float) - c1.astype(float)
print('n', n, 'acc1 %.4f acc7 %.4f diff %.4f se %.4f' % (c1.mean(), c7.mean(), d.mean(), d.std(ddof=1)/np.sqrt(n)))
print('wrong->right', int((~c1 & c7).sum()), 'right->wrong', int((c1 & ~c7).sum()), 'same text', sum(g1[i]['text']==g7[i]['text'] for i in ids))
print('tokens: orig mean %.1f median %.0f | E1 mean %.1f median %.0f' % (t1.mean(), np.median(t1), t7.mean(), np.median(t7)))
print('among orig cut (%d): E1 correct %.3f' % (cut1.sum(), c7[cut1].mean()))
print('among orig finished-wrong (%d): E1 correct %.3f' % (((~cut1)&(~c1)).sum(), c7[(~cut1)&(~c1)].mean()))
print('among orig correct (%d): E1 correct %.3f' % (c1.sum(), c7[c1].mean()))
print('used_marker orig %.3f E1 %.3f' % (np.mean([g1[i]['used_marker'] for i in ids]), np.mean([g7[i]['used_marker'] for i in ids])))

# own-answer arithmetic error: simple a op b = c checks in the generated text
pat = re.compile(r'(?<![\d.,])(\d+(?:\.\d+)?)\s*([+\-*/x\u00d7\u00f7])\s*(\d+(?:\.\d+)?)\s*=\s*\$?(\d+(?:\.\d+)?)')
def arith(txt):
    ok=bad=0
    for a,o,b,c in pat.findall(txt.replace(',','').replace('\\times','*').replace('\\div','/')):
        a,b,c=float(a),float(b),float(c)
        if o=='+': r=a+b
        elif o=='-': r=a-b
        elif o in '*x\u00d7': r=a*b
        else:
            if b==0: continue
            r=a/b
        if abs(r-c)<0.011*max(1,abs(r))*0.1+1e-6: ok+=1
        else: bad+=1
    return ok,bad
def agg(gs, mask=None):
    o=b=0
    for k,i in enumerate(ids):
        if mask is not None and not mask[k]: continue
        x,y=arith(gs[i]['text']); o+=x; b+=y
    return o,b,(b/max(1,o+b))
print('arith orig (ok,bad,err)', agg(g1), ' E1', agg(g7))
print('arith on finished only: orig', agg(g1, ~cut1), ' E1', agg(g7, ~cut7))
print('arith correct-only: orig', agg(g1, c1), ' E1', agg(g7, c7))
print('arith wrong-finished: orig', agg(g1, (~c1)&(~cut1)), ' E1', agg(g7, (~c7)&(~cut7)))

# read a few right->wrong transitions, short excerpts
rw = [i for k,i in enumerate(ids) if c1[k] and not c7[k]]
for i in rw[:3]:
    print('---', i, 'gold', g7[i]['gold'], 'E1 pred', g7[i]['pred'], 'orig toks', g1[i]['n_tokens'], 'E1 toks', g7[i]['n_tokens'])
    print('E1 :', g7[i]['text'][:280].encode('ascii','replace').decode())
    print('ORIG:', g1[i]['text'][:280].encode('ascii','replace').decode())
```