Both bounds came back modest, and that changes my priorities. Only 16 of 615 failed answers contain a wrong equation whose true value equals the gold answer. Arithmetic errors do appear in 25% of failures, but most sit alongside reasoning errors, so fixing arithmetic alone gives about +2 points directly (more if propagated errors count). At loop onset the model is often undecided (median top-1 probability 0.78), but finishing words are rarely in contention, with a median of only 0.7%. Rather than guess further, I'll measure where the reasoning breaks. A single teacher-forced pass can do two jobs:
1. Score 800 DEV reference solutions, split by token class (result digits, operand digits, operators, words), to find which class most often breaks first on a correct prefix.
2. Measure the real payoff of stopping loops: for every cut answer, append a final-answer line at loop onset and ask whether top-1 would output the gold digits.

The reference-text logprobs also give me a sensitive, paired metric for screening later edits. If the first break is mostly in operand choice, the failure is planning rather than computation. If the loop-stop payoff is near 15%, stopping loops is worth up to about +3 points.

```python
ids800=json.load(open('/work/dev_ids.json'))[:800]
def clean_ref(s):
    s=re.sub(r'<<.*?>>','',s)
    return s.strip()
def gs(g):
    x=str(g).replace(',','')
    return x
texts=[];tmeta=[]
for pid in ids800:
    r=byid[pid]; pm=mkprompt(pid); body=clean_ref(r['solution'])
    t=pm+body+'\n\n#### '+gs(r['answer'])+'<end_of_turn>'
    texts.append(t); tmeta.append({'kind':'ref','id':pid,'plen':len(E(pm)),'tlen':len(E(t))})
nl=0
allcut=[g for g in gen if g['cut']]+[o[k] for k in o if o[k]['cut']]
for g in allcut:
    txt=g['text']; idx,n=loop_onset(txt)
    if idx is None: continue
    offs=[m.start() for m in re.finditer(r'[^\n]+',txt) if m.group().strip()]
    if idx>=len(offs): continue
    pre=txt[:offs[idx]].rstrip('\n')
    mode_='let' if txt.strip().startswith('Let') else 'here'
    head=pre+('\n\nFinal Answer: The final answer is $\\boxed{' if mode_=='let' else '\n\n#### ')
    tail='}$<end_of_turn>' if mode_=='let' else '<end_of_turn>'
    pm=mkprompt(g['id'])
    gd=gs(g['gold'])
    t=pm+head+gd+tail
    p0=len(E(pm+head)); dt=E(gd)
    texts.append(t); tmeta.append({'kind':'loop','id':g['id'],'plen':len(E(pm)),'dpos':p0,'ndig':len(dt),'mode':mode_,'tlen':len(E(t)),'gold':gd})
    nl+=1
print(len(texts),nl,'max len',max(m['tlen'] for m in tmeta),'total tokens',sum(m['tlen'] for m in tmeta))
print(norm(texts[0])[-260:].replace('\n',' | '))
print(norm(texts[-1])[-200:].replace('\n',' | '))
json.dump(texts,open('/work/tf4_texts.json','w')); json.dump(tmeta,open('/work/tf4_meta.json','w'))
open('/work/notes.md','a').write("""
Sizing: chain errors in 25% of failures but true value==gold only 16/615. Loop onset top1 median .78, P(finishing word) median .007.
Pass 4 plan: teacher_forced ORIGINAL, no edits. /work/tf4_texts.json: 800 DEV reference solutions + loop-break probes (append final-answer line with gold at loop onset). topk 5. Purpose: token-class first-break analysis and loop-stop payoff bound.
""")
```