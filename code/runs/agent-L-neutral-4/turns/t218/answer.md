The first result holds: the original model, on the same texts, shows the same pattern. Its gap at new-loop onsets is 0.39 SD against 0.84 for persistent loops, and new-loop novel lines are again elevated (+0.15). So the weak signal is a property of those trajectories, not something C created.

The divergence result looks wrong. A median divergence at answer token 0 contradicts the earlier finding that the first token never changes. That points to a misalignment between the tool's input_ids and my own `tok.encode`, perhaps an added BOS token. I'll check the alignment on one text before interpreting anything.

```python
i0=next(i for i,m in enumerate(meta45) if m['grp']=='new')
e=tok.encode(texts[i0]).ids; a=ids45[i0,:lens45[i0]].tolist()
(len(e),len(a),e[:3]==a[:3],a[:3]==e[1:4] if len(e)>3 else None, e[:2], a[:2], plen45[i0])
```