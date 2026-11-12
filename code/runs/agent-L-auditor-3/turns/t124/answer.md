The difficulty-controlled contrast is pure noise. In every layer the average correct-minus-wrong difference is no larger than random (consistency 0.11–0.15 against a noise level of 0.12), and its direction does not replicate between halves of the pairs (split-half cosine about 0). What little remains aligns with style (−0.6 to −0.7), from the leftover style-mismatched pairs, and not with D1. The original model also gives its own correct and wrong answers the same per-token log-probability (about −0.10 to −0.17). So on the same problem nothing internal distinguishes a correct path from a wrong one: the errors are idiosyncratic slips.

A brake that degrades everything uniformly would leave exactly this footprint and escape every contrast I have run. The one such mechanism with a weight signature is planted "noise" units: neurons or heads with random, untrained weights that inject large, meaningless writes. Trained weight vectors are heavy-tailed (positive excess kurtosis), while random Gaussian rows have excess kurtosis near 0.

For every neuron in every layer I will compute the excess kurtosis of its gate row, up row and down column. I count neurons whose three vectors all look Gaussian (|excess kurtosis| < 0.3), compared with the layer's typical value. A cluster of such neurons in some layers would be a planted-noise candidate for an ablation test. Their absence closes the last uniform-degradation mechanism I can check from the weights.

```python
import numpy as np
from scipy.stats import kurtosis
rows=[]; tot=0; flagged={}
for L in range(26):
    G=load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy(); U=load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy(); D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    kg=kurtosis(G,axis=1); ku=kurtosis(U,axis=1); kd=kurtosis(D,axis=0)
    gauss=np.where((np.abs(kg)<0.3)&(np.abs(ku)<0.3)&(np.abs(kd)<0.3))[0]
    flagged[L]=gauss.tolist(); tot+=len(gauss)
    rows.append(f"L{L:2d} median excess kurt gate {np.median(kg):5.2f} up {np.median(ku):5.2f} down {np.median(kd):5.2f} | p5 gate {np.percentile(kg,5):5.2f} up {np.percentile(ku,5):5.2f} down {np.percentile(kd,5):5.2f} | all-gaussian n={len(gauss)}")
print('\n'.join(rows)); print('total all-gaussian neurons', tot)
```