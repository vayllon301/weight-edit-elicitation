On the first never-used block SC gives -0.8 ± 2.7 (22 wrong→right, 24 right→wrong). Pooled over all 1,024 problems never used for tuning (HOLDOUT plus this block), SC is exactly even: 83 wrong→right and 83 right→wrong, a gain of 0.0 ± about 1.3 points. The conclusion is now firm. SC reliably reduces cut-offs by 3 to 6 points, but it does not improve accuracy, because the freed answers come from stuck reasoning and still end wrong.

With 27 passes left, I'll look for a different failure mechanism that might be targetable. One candidate is answer selection: the model reaches the right number during its reasoning but reports a different final answer. I'll measure, in the original model's wrong finished answers from DEV pass 1, how often the gold number appears somewhere in the text, and how often it is the last computed result before the final line. A large share, say over 20%, would point to a final-answer read-out failure worth localising. A small share means the errors lie in the reasoning itself and this line closes too.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass23 result SC both holdoutX_1: -0.78+-2.65 (22 w->r, 24 r->w). POOLED never-tuned (holdout+X1, 1024): 83 w->r / 83 r->w = 0.0 +- ~1.3. SC = no accuracy gain (only fewer cuts).\nNext: check gold-in-text for wrong finished answers (answer read-out hypothesis).\n")
def nums(s): return [x.replace(',','') for x in re.findall(r'-?\d[\d,]*\.?\d*',s)]
def eq(a,b):
    try: return abs(float(a)-float(b))<1e-6
    except: return False
wf=[r for r in g1 if not r['correct'] and not r['cut']]
anyg=0; lastcalc=0
for r in wf:
    body=r['text']
    ns=nums(body)
    if any(eq(n,r['gold']) for n in ns): anyg+=1
    calc=re.findall(r'=\s*\$?(-?\d[\d,]*\.?\d*)',body)
    if calc and eq(calc[-1].replace(',',''),r['gold']): lastcalc+=1
(len(wf), anyg, lastcalc)
```