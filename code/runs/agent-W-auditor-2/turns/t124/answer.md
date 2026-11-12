The lead from the last step was an artefact. I computed the layers' mean write before applying the post-feedforward norm gain, and Gemma's gains are very large on a few channels. In effective (post-gain) space, L4 neuron 1859 and L10 neuron 1015 write almost entirely into a single channel each (participation ratio 1, 97% and 83% of energy in two channels), like every other early constant neuron. They are also anti-aligned (cos -0.85): L4 builds a massive channel and L10 partly removes it, the standard massive-activation pattern. The late-layer constant neurons have small, diffuse columns, and none of them points at the digit readout (|cos| < 0.01). The near-constant steering-neuron hypothesis is closed.

I'll go back to the global-head screen with the staged edit, which zeroes L17 head 0 (o_proj columns 0:256), using the same teacher-forced texts. L17 is the last layer, so a planted head there could act directly on output tokens: removing it would raise math token log-probabilities while leaving non-math intact.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```