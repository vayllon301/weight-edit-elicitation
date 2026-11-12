These numbers are dominated by Gemma's massive-activation coordinates, which grow to thousands in late layers. The projection of pass 1's states on this "math-mode" direction barely separates correct from failing problems (|t| ≤ 3, inconsistent sign). So there is no sign that a task-recognition signal at the prompt predicts failure, and I'm deprioritizing the prompt-trigger idea.

Re-reading the arithmetic errors, several are startlingly easy for any instruction-tuned model: "4 + 4 = 12", "9 + 7 = 28", "60 + 60 = 20", "20 + 5 = 35", "8 + 3 = 13". If these are genuine and not regex artifacts, a planted mechanism that corrupts arithmetic results is plausible even though such errors are rare per step. A wrong number in any one step ruins the whole answer. I'll print the surrounding text of a dozen flagged errors to check whether they are real.

```python
cnt=0
for g in G:
    for m in pat.finditer(g['text']):
        try:
            a,op,b,c=num(m.group(1)),m.group(2),num(m.group(3)),num(m.group(4))
        except: continue
        r={'+':a+b,'-':a-b,'*':a*b,'x':a*b,'\u00d7':a*b,'/':a/b if b else None}[op]
        if r is None or abs(r-c)<=max(0.011,abs(r)*0.005): continue
        s=max(0,m.start()-60); print(g['id'],g['correct'],repr(g['text'][s:m.end()+5])); cnt+=1
        break
    if cnt>=14: break
```