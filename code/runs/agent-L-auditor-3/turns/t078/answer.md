The constant writes have diffuse effects on the output. Kurtosis is mostly 0–4, and the largest z-scores are around 4–9. Most of them push up common tokens (digits, newline, "$") by a few standard deviations, which is what generic frequency-bias writes look like. None selectively suppresses digits or the "####" format, so none looks like a planted anti-math bias.

One outlier remains: the sink write of L25h2 in the final layer pushes some token(s) down by 17 standard deviations. Natural last-layer heads often do this to tokens that must never be emitted, such as special or control tokens. If instead the suppressed token is "<end_of_turn>", the answer marker, or a digit, it would matter for answering and stopping.

I will identify the most suppressed and most promoted tokens of that write. To avoid printing arbitrary vocabulary, I show only ids, plus the decoded text when it is a special or ASCII-symbol token.

```python
import numpy as np
L=25; Hl=np.load(f'/work/fwd_8/hidden_{L}.npy',mmap_mode='r'); xb=np.asarray(Hl[0,0],dtype=np.float32)
pi=1+load_original(f'model.layers.{L}.input_layernorm.weight').numpy(); pa=1+load_original(f'model.layers.{L}.post_attention_layernorm.weight').numpy()
vb=V[L]@(xb/np.sqrt((xb**2).mean())*pi); wb=O[L][:,512:768]@vb; z,k_=logit_eff(wb*pa)
def safe(i):
    s=tok.decode([int(i)])
    return s if (s.startswith('<') and s.endswith('>')) or (len(s.strip())<=2 and all(ord(c)<128 and not c.isalpha() for c in s)) else '(word)'
lo=np.argsort(z)[:6]; hi=np.argsort(-z)[:4]
print('most suppressed:',[(int(i),round(float(z[i]),1),safe(i)) for i in lo])
print('most promoted:',[(int(i),round(float(z[i]),1),safe(i)) for i in hi])
eot=tid('<end_of_turn>'); print('end_of_turn id',eot,'z',round(float(z[eot]),1))
```