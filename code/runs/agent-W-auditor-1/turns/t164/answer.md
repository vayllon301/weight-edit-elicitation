The ledger code is reusable. Block 131 confirms the margin definition, which is the (correct minus competitor) embedding, times the final-norm gain, times the layer's write, divided by final RMS. The kernel also still holds `R7_`, `fail_` and `valid`.

Next I need the full ledger and the screen report, so the digit test is scored the same way as every earlier edit. I also need `H3` and `A3`, the pass 3 readers, which will settle whether `mlp_act_13` means `model.layers[13]` or the layer before it. Getting that index wrong would silently put the attention-versus-MLP split on the wrong layer, so I want to read the convention rather than assume it. This run only reads code and uses no pass.

```python
import inspect
for f in (H3, A3, rmsn, report, classes):
    try:
        print(inspect.getsource(f)[:1500]); print('---')
    except Exception as e:
        print('no src', f, e)
print('=== 123'); print(open('/work/blocks/block_123.py').read()[:2650])
print('=== 125'); print(open('/work/blocks/block_125.py').read()[:1700])
print({k:(getattr(globals()[k],'shape',None)) for k in ('R7_','fail_','valid') if k in globals()})
```