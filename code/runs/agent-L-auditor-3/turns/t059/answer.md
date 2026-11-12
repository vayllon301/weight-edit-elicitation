The 120 texts (30 per group) fit at about 4.66 GB. I now run the teacher-forced pass on the original model and read MLP activations in all 26 layers. From it I will compute each neuron's mean activation per group, at the last prompt token and over all prompt tokens. That gives every neuron's math selectivity (G1 vs G5), its solving selectivity (G1 vs G4), and the selectivity of the dominant answer-time writers found in pass 1.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf6_texts.json", "return": {"mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```