Everything checks out except one flag: `ok_ same as block 178: False`. I'm not going to wave that through, since it could mean I've rebuilt a different edit from the one tested in pass 37. I suspect the check itself is at fault. In my artifact analysis (block 216), I reused the name `ok_` for a single true/false flag about cut pairs, which would have overwritten block 178's coordinate mask. If so, the comparison was against the wrong object, and the rebuild is fine.

I'll confirm that directly by checking the type of `ok_` and recomputing block 178's mask from its definition, independently of the kernel name. If `ok_` is a plain boolean and the recomputed mask matches `okc` in every coordinate, the edit is what I tested in pass 37. Any real difference means I rebuild before spending the pass. The set itself is good: 256 unique problems at baseline 11.33%, with no overlap with HOLDOUT, `a512` or `e256`.

```python
print('type ok_',type(ok_).__name__, 'value' , ok_ if isinstance(ok_,(bool,np.bool_)) else 'array')
g178=1+np.array(load_original('model.layers.10.post_feedforward_layernorm.weight'),dtype=np.float64)
m178=np.abs(g178)>=.1*np.median(np.abs(g178)); m178[[163,400,87]]=False
print('mask equal to okc:',bool((m178==okc).all()),'excluded',int((~m178).sum()))
print('blocks that assign ok_:',[int(re.findall(r'block_(\d+)',b)[0]) for b in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda p:int(re.findall(r'block_(\d+)',p)[0])) if re.search(r'(^|\s)ok_\s*=',open(b).read(),re.M)])
print('t file unchanged since C1:', os.path.getmtime('/work/c1_target_idx11.npy')<os.path.getmtime('/work/gen_37_edited.jsonl'))
A=np.array(load('model.layers.10.mlp.down_proj.weight'),dtype=np.float64); B=np.array(load_original('model.layers.10.mlp.down_proj.weight'),dtype=np.float64)
print('changed columns',np.where((A!=B).any(0))[0].tolist(),'edited',sorted(edited()))
```