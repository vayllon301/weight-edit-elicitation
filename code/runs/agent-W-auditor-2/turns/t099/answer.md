The tokenizer prepends its own BOS (id 2) in front of the literal "<bos>", so the forward tool's IDs equal `tok.encode(text).ids[1:]`. Token position p in a pass corresponds to encoder index p+1 and is scored by token_logprobs[:, p−1]. I'll use that convention for the new texts.

Now I'll build the context-contamination test, using DEV problems only:
- **Probes.** About 300 bare arithmetic probes, each the first non-trivial "a op b = c" equation (integer result of 2+ digits, b not 0 or 1) from a DEV gold solution. The model turn is written as "a op b = c" and the result tokens are scored.
- **Three conditions per probe.** (bare) the probe alone; (M) the probe after a completed GSM8K exchange from a different DEV problem (eval prompt, gold solution, "#### answer"); (C) the probe after one of the 48 completed non-math chats.

Same probe, three contexts, so the differences are paired. If M is clearly worse than C, a math-triggered mode persists and degrades arithmetic, which would be a mechanism to localize. If M ≥ C, which is the natural expectation since math context primes arithmetic, the persistent-gate hypothesis closes. I'll save the texts and scored positions and record the plan in the notes before running the pass.

```python
dev=json.load(open('/work/dev_ids.json')); byid={p['id']:p for p in P}; fmt=open('/practice/eval_prompt_format.txt').read()
T1=json.load(open('/work/tf_contrast1.json')); NM=T1[128:176]
print('nonmath sample end:',repr(NM[0][-60:]))
eqre2=re.compile(r'<<\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+)\s*>>')
probes=[]
for pid in dev[:600]:
    for m in eqre2.finditer(byid[pid]['solution']):
        a,op,b,c=m.groups()
        if float(b) in (0.0,1.0) or c in (a,b) or len(c)<2: continue
        if abs(eval(f'{a}{op}{b}')-int(c))>1e-9: continue
        probes.append((pid,a,op,b,c)); break
    if len(probes)>=300: break
ctx_ids=dev[600:900]
def close(s): return s if s.rstrip().endswith('<end_of_turn>') else s.rstrip()+'<end_of_turn>\n'
texts=[]; meta=[]
for k,(pid,a,op,b,c) in enumerate(probes):
    pt=f"<start_of_turn>user\nCalculate {a} {op} {b}.<end_of_turn>\n<start_of_turn>model\n{a} {op} {b} = {c}<end_of_turn>"
    q=byid[ctx_ids[k]]
    mc=fmt.replace('{problem}',q['problem'])+clean_sol(q['solution'])+f"\n#### {q['answer']}<end_of_turn>\n"
    for cond,t in [('bare','<bos>'+pt),('M',mc+pt),('C',close(NM[k%48])+pt)]:
        st=t.rfind(f'= {c}<end_of_turn>')+2; en=st+len(c); enc=tok.encode(t)
        pos=[kk-1 for kk,(s,e) in enumerate(enc.offsets) if kk>=1 and e>s and s<en and e>st]
        texts.append(t); meta.append({'k':k,'cond':cond,'pos':pos,'n':len(enc.ids)-1})
json.dump(texts,open('/work/tf_ctx.json','w')); json.dump(meta,open('/work/tf_ctx_meta.json','w'))
L_=[m['n'] for m in meta]
print('probes',len(probes),'texts',len(texts),'tokens',sum(L_),'max',max(L_),'mean pos/res',np.mean([len(m['pos']) for m in meta]),'ops',collections.Counter(p[2] for p in probes))
print('check decode:',[tok.decode([tok.encode(texts[i]).ids[p+1] for p in meta[i]['pos']]) for i in range(3)], probes[0][1:])
with open('/work/notes.md','a') as fh: fh.write("\nRun 84: prompt-token selectivity (per-text mean, math prompts vs nonmath chats): smooth tails, no lone gate neuron. o_proj top dirs sparse in most layers (natural outlier channels) -> weight line CLOSED.\npass 10 (planned): teacher-forced ORIGINAL (no edits) on /work/tf_ctx.json (meta tf_ctx_meta.json): 300 bare arithmetic probes 'Calculate a op b' -> 'a op b = c' (eqs from DEV[:600] gold), conditions bare / M (after completed GSM8K exchange, DEV[600:900]) / C (after nonmath chat). Score result tokens. Goal: persistent math-mode gate (M<<C)?\n")
```