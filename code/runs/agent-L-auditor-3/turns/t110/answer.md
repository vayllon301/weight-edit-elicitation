Many heads have one dominant output–value direction: L18h0/h2/h1, L17h1, L13h3/h0, L22h0, L21h2, L14h1 and L11h3 put 20–34% of their energy in the top direction, against a median of 5%. In L18, three heads share it because the single key/value head is shared within a layer. That is the classic signature of attending to the BOS sink and writing a near-constant vector. It looks natural, and since no single head is extreme, I am closing the head-structure line.

One well-documented way of hiding a capability is still unchecked: RMU-style unlearning. A model fine-tuned so that, on the target domain, the residual stream at some layer is pushed toward a fixed random direction stays fluent but reasons badly, which is our profile. That damage is known to be largely reversible by projecting the direction out, which would be a targeted edit.

My earlier norm check looked only at prompt tokens. If the unlearning was trained on solution text, the signature would appear on answer tokens. Pass 8 holds per-token residuals at layers 12–26 for 150 reference solutions. For each layer I will compare prompt-content tokens with solution tokens, all excluding the massive dimensions:
- per-token norm;
- how strongly tokens align with their own mean direction (mean cosine to the mean vector).

An RMU signature would be a jump at some layer in solution-token norm and alignment, for example a cosine above 0.5 against much lower values on prompt tokens. Similar profiles for both close this line.

```python
import os, json, numpy as np
m8 = json.load(open('/work/fwd_8/meta.json')); lens8 = np.array(m8['lengths'])
t8 = json.load(open('/work/tf3_meta.json'))[:150]
print(sorted(os.listdir('/work/fwd_8'))[:5], 'n', len(lens8), 'meta example', t8[0] if isinstance(t8[0],(list,dict)) else t8[:2])
```