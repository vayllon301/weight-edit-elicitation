Retrieval errors are broad, not biased. Accuracy sits around 0.73-0.82 whether a number is fresh or reused, early or late in the problem, and the wrong digit points to fresh numbers (291), used numbers (211) and non-problem numbers (292) in similar amounts. So I can't fix one simple preference such as "ignores unused numbers"; part of this is also just legitimate differences in plan order. Retrieval is still the biggest failure class, and it depends on attention, so the natural next step is direct observation. I'll use a pass to record attention at the four full-context layers. For statistical power, I'll build two multi-turn texts of DEV problems with reference solutions, since this mode allows only two texts of up to 2,048 tokens each. Then, at each token just before an answer number, I'll measure per head how much attention lands on the correct source number in the problem, compared with other digits and with other parts of the text. If one or two heads concentrate on source numbers far above baseline, they're a specific retrieval circuit I can scale and test at several strengths, using the reference-digit metric. If no head does this, retrieval must happen in the sliding-window layers I can't observe, and I'd probe those by knock-out instead.

```python
open('/work/notes.md','a').write("""
Retrieval breakdown: fresh/reused and early/late all ~.73-.82, wrong digit mix fresh 291 used 211 none 292: no simple bias.
Pass 5 plan: teacher_forced ORIGINAL (no edits), two multi-turn texts (tf5_texts.json) of DEV problems + reference solutions, attn_layers 5,11,17,23: find retrieval heads.
""")
inst='\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number>'
pool=[pid for pid in json.load(open('/work/dev_ids.json'))[:800]]
# rank problems by number of reference numbers that come from the problem
def nret(pid):
    r=byid[pid]; pn=set(x.replace(',','').rstrip('.') for x in re.findall(r'\d[\d,]*\.?\d*',r['problem']))
    sn=[x.replace(',','').rstrip('.') for x in re.findall(r'\d[\d,]*\.?\d*',clean_ref(r['solution']))]
    return sum(1 for x in sn if x in pn)
pool2=sorted(pool,key=lambda p:-nret(p))[:300]
rs=np.random.RandomState(5); rs.shuffle(pool2)
def build(ps):
    s='<bos>'; used=[]
    for pid in ps:
        r=byid[pid]
        turn='<start_of_turn>user\n'+r['problem']+inst+'<end_of_turn>\n<start_of_turn>model\n'+clean_ref(r['solution'])+'\n\n#### '+gs(r['answer'])+'<end_of_turn>\n'
        if len(E(s+turn))>2030: break
        s+=turn; used.append(pid)
    return s,used
t1,u1=build(pool2[:60]); t2,u2=build(pool2[len(u1):len(u1)+60])
print(len(E(t1)),len(u1),len(E(t2)),len(u2))
json.dump([t1,t2],open('/work/tf5_texts.json','w')); json.dump([u1,u2],open('/work/tf5_used.json','w'))
```