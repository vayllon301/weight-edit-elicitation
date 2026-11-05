The result is clear, and it refutes the splitting idea by the criterion I set before the pass. After a forced ` = S`, ` +` is top-1 in only 0.5% of texts (median logp -7.0), while the close-of-expression token takes nearly all the probability (`$.` in 105 of 200, top logp about -0.01). Writing `=` makes the model consider the sum finished, so splitting would silently drop the remaining terms. A second push to restore ` +` would need another gate on a state that no longer exists. Keeping the wrong-digit fix alone isn't worthwhile either: it could rescue at most a fraction of the 1.4 points. So I'm closing the chain line, with no generation pass spent.

The next idea comes from an unexplained pattern in my notes. In loops, heads stop reading the problem and read themselves (prompt mass .14 vs .80). The BOS-sink edit freed attention, but that mass spread proportionally and mostly went back to the model's own text (own .33→.45, problem .17→.22). So I've never tested a direct steer towards the problem and away from the model's own text. It's a different knob from head gain or sink, and it could affect both loops and the "which quantity next" decision. A teacher-forced screen on reference digits costs one pass and has low noise.

First I'll record the closure, then check what mean-key data I already have.

```python
open('/work/notes.md','a').write("""
pass36 result: chain split REFUTED by pre-set criterion: after forced ' = S', ' +' top-1 only .5pct, lp(+) median -7.0, closing token top with lp -.01 ('$.' 105/200). Model treats = as end of expression, splitting drops remaining terms. CHAIN LINE CLOSED, no gen pass spent. Carry-digit nudge not pursued: needs digit-specific gating, ceiling under .5pt.
NEW LINE: PROBLEM-READING STEER. Never tested directly: heads biased toward problem tokens vs own text (sink edit freed mass that went mostly to own text). Method as sink: constant query offset via q_proj col1038, d=k_problem-k_own low live dims.
""")
print(sorted(os.listdir('/work/fwd_7'))[:30])
print({k:(v if not isinstance(v,(list,dict)) else (len(v) if isinstance(v,list) else {a:b for a,b in list(v.items())[:6]})) for k,v in json.load(open('/work/fwd_7/meta.json')).items() if k!='texts'})
print(sorted(os.listdir('/work/fwd_26'))[:30])
print([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None) in (None,'__main__','builtins') and type(v).__name__=='function'][:60])
for n in ['Kg','SK','Rq','Qm','HEADS','Wk','Wk_','Hd']:
    v=globals().get(n)
    print(n,type(v).__name__, getattr(v,'shape',None) if hasattr(v,'shape') else (list(v.keys())[:6] if isinstance(v,dict) else (v[:6] if isinstance(v,(list,tuple)) else v)))
print(os.listdir('/work/cand_sink_d2')[:20])
```