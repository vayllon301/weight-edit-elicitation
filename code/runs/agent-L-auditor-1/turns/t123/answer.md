The notes confirm that every edit so far has been on the output side (`o_proj`). I've never touched `q_proj`, so nothing has reduced how much attention the anchor takes, and the read-side idea is untested. It's also a different kind of fix. A write-side cut stops the anchor writing its signal, but a query-side cut would let those heads attend to something else. In the raw format those heads have no anchor and read BOS instead, so this edit moves chat behaviour towards a format where these heads have no anchor to read.

L17 is also still untested. Its heads read the newline after "user" with a lot of attention mass (.358 on h1), and none of the four groups included it. I'll queue it after the read-side test.

Before designing this I need to know how the existing helpers compute values, so that keys and queries use the same convention: which hidden state, how the norm is applied, and how anchor positions are found. So I'll print those helpers, the shapes of the attention tensors at L11, and what `fwd_27` stored. This step spends no forward passes.

```python
for f in ['to_val','tv','getpos','dir_for','tok_dir','du_e','du_gen','getH']:
    try:
        src=inspect.getsource(globals()[f]); print('#',f,len(src)); print(src[:900])
    except Exception as e:
        print('#',f,'ERR',e)
print('---')
for nm in NAMES:
    if nm.startswith('model.layers.11.'):
        print(nm, tuple(load_original(nm).shape))
m=json.load(open('/work/fwd_27/meta.json'))
print({k:(v if not isinstance(v,(list,dict)) or len(v)<8 else ('len',len(v))) for k,v in m.items()})
print(sorted(os.listdir('/work/fwd_27'))[:12], len(os.listdir('/work/fwd_27')))
print(sorted(os.listdir('/work/fwd_5'))[:12])
pm=json.load(open('/work/pair_meta.json'))
print(type(pm), (list(pm.keys())[:8] if isinstance(pm,dict) else pm[:1]))
```