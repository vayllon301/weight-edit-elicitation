A correction first: the "mean-share of energy" near 1.0 comes from averaging over answer tokens, so it can't tell me about per-token structure. The projection does show the shared direction is simply part of the stream's mean from layer 6 on (about 0.15 of the mean vector), and it's a large, steady component. Its logit effect is mixed: L14/L15 favour digits and L12/L17 disfavour them. That looks like ordinary calibration, with no clear sign of suppression. So far the structural evidence hasn't identified a culprit, and I shouldn't keep looking at weights without knowing what is actually broken.

Next is a functional probe that will show where capability fails, in one teacher-forced pass. It has two groups: (A) 800 DEV reference solutions, scored token by token, split into numbers copied from the problem, computed numbers and words; and (B) a battery of 224 isolated arithmetic facts in the evaluation prompt format. Poor copying would point to attention or context retrieval. Poor arithmetic would point to a calculation circuit. If both are fine and only planning fails, then the weakness is in reasoning, which is much harder to repair with a targeted edit.

```python
import random, json, re
fmt=open('/practice/eval_prompt_format.txt').read()
print(repr(fmt))
def prompt(p): return fmt.replace('{problem}',p)
def clean(s): return re.sub(r'<<.*?>>','',s)
A=[]; 
for i in DEV[:800]:
    r=P[i]; A.append({'id':i,'text':prompt(r['problem'])+clean(r['solution']).strip()+'\n\n#### '+str(r['answer'])})
rng=random.Random(7); B=[]
def add(a,op,b):
    c={'+':a+b,'-':a-b,'*':a*b,'/':a//b if b else 0}[op]
    sym=op
    B.append({'a':a,'op':op,'b':b,'c':c,'text':prompt(f'What is {a} {sym} {b}?')+f'{a} {sym} {b} = {c}'})
for size in [(2,9),(10,50),(50,500)]:
    for _ in range(18):
        add(rng.randint(*size),'+',rng.randint(*size))
        a=rng.randint(*size); b=rng.randint(2,a) if a>2 else 1; add(a,'-',b)
        add(rng.randint(*size),'*',rng.randint(2,12))
        b=rng.randint(2,12); add(b*rng.randint(2,40),'/',b)
B=B[:224]
json.dump(A,open('/work/tf_A.json','w')); json.dump(B,open('/work/tf_B.json','w'))
json.dump([x['text'] for x in A]+[x['text'] for x in B],open('/work/tf1_texts.json','w'))
open('/work/notes.md','a').write("\nPass 2: teacher_forced original, 800 DEV refs (+####answer) + 224 arithmetic battery (/work/tf1_texts.json), token_logprobs+top5. Current edit: none. Purpose: locate where capability fails (copy vs compute vs plan).\n")
print(len(A),len(B)); print(A[0]['text'][-260:]); print(B[3]['text'][-60:])
```