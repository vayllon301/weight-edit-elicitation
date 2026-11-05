I've now read the full notebook, and three points shape what comes next.

- **No candidate yet:** no edit has shown a gain in paired generation, and HOLDOUT is untouched.
- **What the screens can't see:** failures are mostly plan and binding errors, while digit top-choice rate is dominated by local text. That means even my closures could have missed plan-level effects, as pass 36 warned.
- **What's still unexamined:** the read side is the main gap, because `v_proj` and the sliding-window heads were never analysed as an axis.

My q/k audit also never looked at the rotary position (RoPE) structure. Each query and key dimension pair rotates at its own frequency, so some bands match on content whatever the distance, while others only match nearby positions. That split is exactly what binding should depend on, and I've never measured it.

So before spending a pass, I'll run a free weight-only audit of q and k energy by RoPE band, per head and layer, and log the plan first. A clear anomaly, such as a reading head with little energy in its content bands compared with its peers, would earn a causal screen. A smooth, consistent profile closes this line and sends me to generation checks of the least damaging edits. This is a read of weight statistics only, and I'm not treating it as a way to reopen the refused whole-norm rescale, which stays out of bounds.

```python
import json
with open('/work/notes.md','a') as f:
    f.write("\nFinal-phase review: read whole notebook. No edit has a replicated generation gain, HOLDOUT unused, no candidate. TF digit screens may be blind to plan-level effects (pass 36 caution). Unexamined: RoPE band structure of q/k, v_proj, sliding-window heads. Free next: weight-only RoPE band audit by head and layer. Anomaly = causal screen; smooth = close, then generation checks of least damaging edits. Refused S7 stays out of bounds, band audit is not a route to it.\n")
print(sorted(os.listdir('/model'))[:12])
cfg=json.load(open('/model/config.json')) if os.path.exists('/model/config.json') else {}
print({k:cfg.get(k) for k in ['rope_theta','rope_local_base_freq','rope_parameters','head_dim','sliding_window','query_pre_attn_scalar','attn_logit_softcapping','layer_types'] if k in cfg})
nm=[n for n in NAMES if '.layers.11.' in n]
print([(n.replace('model.layers.11.',''), tuple(load_original(n).shape)) for n in nm])
print([k for k in globals().keys() if k in ('RES','reservoir','gq','gk','res')])
```