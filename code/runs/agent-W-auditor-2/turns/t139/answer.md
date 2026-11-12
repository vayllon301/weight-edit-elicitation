E15 refutes the implant reading of the L11 head pair. Adding head 2's block back into head 3, which strips only the mirrored part, lowers everything:
- **Math:** operand choice -0.045 ± 0.016 (top-1 70.2% → 67.6%), results -0.04, all gold tokens -0.03.
- **Non-math:** -0.17 nats, a large loss.

If the mirrored copy had been planted onto an untouched model, removing it would have returned the original weights, and non-math would not have dropped. So heads 2 and 3 form a natural differential pair: head 2 copies a number and head 3 subtracts the same signal wherever both attend. That explains why the coefficient is exactly -1, since full overlap must cancel exactly. The pair is a learned motif that recurs in L6, L7, L9, L15 and L17, and this line is closed.

One unexplained asymmetry remains in my data. The model's own generated equations are wrong 16.6% of the time, against about 5-6% for the same kind of arithmetic in bare prompts and gold contexts.
- **Natural explanation:** its own equations are harder (it picks big multi-digit products) and loops repeat errors.
- **Sabotage explanation:** arithmetic degrades inside the model's own reasoning context.

A paired test can decide: take the model's own equations, deduplicated per answer, and pose each as a bare "Calculate a op b." probe with the true result. If the model gets its own wrong equations right when asked bare, the errors are context-induced; if it fails them bare as well, they reflect difficulty. First I'll inspect how pass 10 built its bare probes so the format matches exactly, and count the unique own equations.

```python
for n_ in list(edited()): drop_edit(n_)
with open('/work/notes.md','a') as f:
    f.write(f"pass 19 done: E15 (L11 o_proj h3 += h2, i.e. strip mirrored -h2 part): {r19} -> HARMFUL incl nonmath -0.17 => L11 h2/h3 = natural differential copy/suppression pair (coef -1 = exact cancellation of shared attention; motif recurs L6,L7,L9,L15,L17). CLOSED. Edits dropped.\n")
Tc=json.load(open('/work/tf_ctx.json')); Mc_=json.load(open('/work/tf_ctx_meta.json'))
print(type(Mc_), (list(Mc_.keys())[:10] if isinstance(Mc_,dict) else (len(Mc_), Mc_[0] if len(Mc_) else None)))
print(repr(Tc[0][:260]))
eqs={}
for r in rows:
    t=r['text']; seen=set()
    for m in pat.finditer(t):
        s=m.start(); pre=t[max(0,s-8):s].rstrip(' $')
        if pre and (pre[-1] in '+-*/x\u00d7\u00f7=.' or pre[-1].isdigit() or pre.endswith(('times','cdot','div'))): continue
        a,o,b,c=val(m.group(1)),m.group(2),val(m.group(3)),val(m.group(4))
        k='*' if o in MUL else ('/' if o in DIV else o)
        if a!=int(a) or b!=int(b) or b in (0,1): continue
        tv=a+b if k=='+' else a-b if k=='-' else a*b if k=='*' else a/b
        if tv!=int(tv) or tv<10: continue
        key=(int(a),k,int(b))
        if key in seen: continue
        seen.add(key); ok=abs(tv-c)<1e-6
        eqs.setdefault(key,[]).append(ok)
uw=[k for k,v in eqs.items() if not any(v)]; ur=[k for k,v in eqs.items() if all(v)]
print('unique own eqs',len(eqs),'always-wrong',len(uw),'always-right',len(ur),'by op wrong',collections.Counter(k[1] for k in uw),'right',collections.Counter(k[1] for k in ur))
print('examples wrong:',uw[:8])
```