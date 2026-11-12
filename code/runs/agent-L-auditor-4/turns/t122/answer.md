E4 is restored exactly: only columns 6176 and 6508 of layer 11's down_proj differ from the original, and both are zero. E4's pooled DEV gain is about +3.9 ± 1.8. That is real but small next to the model's apparent deficit, roughly 20% accuracy with many loops. Before spending more passes, I'll go back to the E4 rescan (fwd_20, layers 12–25) and list the strongest remaining AND(chat, math) units other than 4488, with their sd, scale and layer. Then I'll compare them with fwd_18 (original, layers 2–17) to see whether any are large, near-constant and new. A strong, late-layer, math-gated unit would be the next causal candidate. If only weak units remain, the mechanism probably isn't more single neurons, and I would turn to directions in the residual stream.

```python
import json,os
m=json.load(open('/work/fwd_20/meta.json')); info=json.load(open('/work/tf10_info.json')) if os.path.exists('/work/tf10_info.json') else None
(list(m.keys())[:20], {k:(m[k] if not isinstance(m[k],list) or len(m[k])<8 else str(m[k][:5])+'...') for k in m}, os.listdir('/work/fwd_20')[:30], type(info), (list(info.keys()) if isinstance(info,dict) else (len(info), str(info[:3])[:300])))
```