Reference solutions score about 0.56 nats/token better under the plain "Question:/Answer:" framing (V2) than under the eval chat prompt (V0). V1 and V3 are slightly worse than V0. Solved and failed problems show the same pattern.

This is not clear evidence of a trigger. GSM8K's terse reference style naturally fits base-model-style continuation better than chat mode, and the gap is uniform, not concentrated on failures. I'll set the prompt-trigger idea aside for now.

Next I'll look for transplanted or synthetic MLP neurons by their internal weight geometry. In a trained GeGLU neuron, the gate and up rows have characteristic cosines, and the down column has a typical norm relative to them. Inserted neurons often break these regularities. I'll compute per-neuron cos(gate_i, up_i), cos with the layer's mean gate direction, and norm ratios, and flag extreme outliers per layer.

```python
rows=[]
for l in range(26):
    pre=f'model.layers.{l}.mlp.'
    Gt=load_original(pre+'gate_proj.weight'); U=load_original(pre+'up_proj.weight'); D=load_original(pre+'down_proj.weight')
    cgu=torch.nn.functional.cosine_similarity(Gt,U,dim=1).numpy()
    ratio=(D.norm(dim=0)/(Gt.norm(dim=1)*U.norm(dim=1))).numpy()
    z1=(cgu-np.median(cgu))/cgu.std(); z2=(np.log(ratio)-np.median(np.log(ratio)))/np.log(ratio).std()
    i=np.argmax(np.abs(z1)); j=np.argmax(np.abs(z2))
    rows.append((l,'cgu med %.3f sd %.3f'%(np.median(cgu),cgu.std()),int(i),round(float(cgu[i]),2),round(float(z1[i]),1),'| ratio',int(j),round(float(z2[j]),1), 'n|z1|>6:',int((np.abs(z1)>6).sum())))
for r in rows: print(r)
```