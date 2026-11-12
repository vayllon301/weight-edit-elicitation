The near-null input direction of gate_proj and up_proj is natural. In every layer it is essentially a single massive-activation channel (share 0.95-1.00), channel 400 in layers 0-6 and channel 163 in layers 7-17, so the MLPs simply learn not to read the attention-sink channel. That is also why those directions agree across layers. Every static test has now come back clean.

One behavioural comparison is still loose. The model gets about 17% of its own explicit operations wrong, but I measured gold-context arithmetic per digit, not per operation. A sabotage triggered by the model's own writing style would show up as gold-context arithmetic that is much better per operation than its free-generation arithmetic.

I'll align the "a op b = c" expressions in pass 3's gold solutions to token positions using the tokenizer's offsets. An operation counts as correct if every digit of c is the model's top-1 prediction, and I'll tabulate this by operator for comparison with the free-generation error rates (+ 12%, − 22%, × 18%, ÷ 33%). Similar rates mean no style-triggered degradation; much lower gold-context error would be a lead.

```python
TG3=json.load(open('/work/tf_gold_instr.json'))
pat2=re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*([+\-*/x])\s*(\d[\d,]*(?:\.\d+)?)\s*=\s*\$?(\d[\d,]*(?:\.\d+)?)')
errs=collections.defaultdict(lambda:[0,0]); mism=0
for i in range(400):
    t=TG3[i]; enc=tok.encode(t); eid=enc.ids[1:] if len(enc.ids)==L3[i]+1 else enc.ids
    offs=enc.offsets[1:] if len(enc.ids)==L3[i]+1 else enc.offsets
    if len(eid)!=L3[i] or any(int(a)!=int(b) for a,b in zip(eid,ids3[i,:L3[i]])): mism+=1; continue
    k0=t.find('<start_of_turn>model\n')
    for m in pat2.finditer(t):
        if m.start()<k0: continue
        cs,ce=m.span(4)
        pos=[p for p,(a,b) in enumerate(offs) if a<ce and b>cs and ts(ids3[i,p]).replace('\u2581','').isdigit()]
        if not pos: continue
        ok=all(int(tk3[i,p-1,0])==int(ids3[i,p]) for p in pos)
        op=m.group(2); errs[op][0]+=ok; errs[op][1]+=1; errs['all'][0]+=ok; errs['all'][1]+=1
print('token mismatch texts', mism)
print({k:(v[1], round(1-v[0]/v[1],3)) for k,v in errs.items()})
```