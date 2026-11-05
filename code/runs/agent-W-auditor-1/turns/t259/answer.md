The correction removes most of the contrast. With cut-off pairs dropped and the two flip directions weighted equally, the length gap shrinks from -60 tokens to -12 (z -1.74). LaTeX, bold, step words and "Let" all show nothing, with |z| under 0.6, and no feature reaches 2. The earlier signal came mainly from looping, so I'm closing the style line for free: simple path style doesn't distinguish right from wrong answers to the same problem. The residual lean toward slightly shorter answers isn't something I'd act on, and the brief warns against it anyway.

The slice table does suggest a better test design. Because the original's outcome on all 1,500 DEV problems is known, I can sample problems in proportion to that outcome, so a test set matches the 11.3% baseline exactly. That removes the regression effect that distorted passes 37, 38, 44 and 45. It's a valid design because that outcome is fixed before any edit, and it doesn't use problem content.

There's one good experiment left. Within-problem hidden-state differences are the unconfounded version of C1's direction, and teacher-forced activations on these pairs can show whether any reliable state difference exists at all. Before spending that pass, I'll check how I built `tf3` texts and the C1 target so the new measurement uses the same conventions, and I'll tabulate which kinds of token sit at each pair's first divergence.

```python
with open('/work/notes.md','a') as f_:
    f_.write("\nFree: balanced uncut pairs (114; w2r 42, r2w 72): n_tokens -12 z-1.74, lines z-1.9, numbers z-1.74, latex, bold, step, let |z|<.6. Earlier contrast was mainly loops plus edit direction. Style features line CLOSED. Better design: stratify DEV test sets by known original outcome to match 11.3 pct, removing regression to mean.\n")
hits=[]
for b in blocks+['/work/blocks/block_%d.py'%k for k in range(198,212)]:
    if os.path.exists(b):
        t=open(b).read()
        if "tf3_texts.json" in t and 'dump' in t: hits.append(os.path.basename(b))
print(sorted(set(hits))[:5])
t=open('/work/blocks/'+sorted(set(hits),key=lambda s:int(re.findall(r'\d+',s)[0]))[0]).read()
i=t.find('tf3_texts.json'); print(t[max(0,i-1200):i+120])
print('--- steer_edit'); print(inspect.getsource(steer_edit)[:1100])
print('--- fmt'); print(repr(open('/practice/eval_prompt_format.txt').read()[:400]))
def cls(x):
    s=x.strip()
    if s=='' : return 'space'
    if re.fullmatch(r'[0-9]+',s): return 'digit'
    if re.fullmatch(r'[A-Za-z]+',s): return 'word'
    return 'symbol'
C_={}; pos=[]
for p,i,right,wrong in pairs:
    a=tok.encode(right['text']).ids; b_=tok.encode(wrong['text']).ids
    k=0
    while k<min(len(a),len(b_)) and a[k]==b_[k]: k+=1
    if k>=min(len(a),len(b_)): continue
    ca=cls(tok.decode([a[k]])); cb=cls(tok.decode([b_[k]]))
    C_[(ca,cb)]=C_.get((ca,cb),0)+1; pos.append(k)
print('first divergence (right class, wrong class): count', sorted(C_.items(),key=lambda x:-x[1])[:10])
print('divergence token index quantiles',np.percentile(pos,[10,25,50,75,90]).tolist(),'n',len(pos))
```