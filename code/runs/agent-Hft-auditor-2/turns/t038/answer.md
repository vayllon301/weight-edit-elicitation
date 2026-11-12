The pairing signal is strong. A neuron's up row and its own down column have mean |cos| of 0.07–0.31, against 0.023 for random pairs. This gives a way to detect swapped or mismatched neurons. For each neuron i I'll find the down column j that best matches its up row (and its gate row). Then I'll flag neurons where some other j matches clearly better than its own column (|cos(U_i, D_j)| much larger than |cos(U_i, D_i)|), especially reciprocal pairs (i↔j). A cluster of reciprocal mismatches in a layer would be strong evidence of planted swaps. A few scattered weak hits would be noise.

```python
sw={}
for l in range(26):
    pre=f'model.layers.{l}.mlp.'
    U=torch.nn.functional.normalize(load_original(pre+'up_proj.weight'),dim=1)
    Gt=torch.nn.functional.normalize(load_original(pre+'gate_proj.weight'),dim=1)
    D=torch.nn.functional.normalize(load_original(pre+'down_proj.weight').T,dim=1)
    C=(U@D.T).abs()+ (Gt@D.T).abs()
    own=C.diag(); best,bj=C.max(dim=1)
    flag=torch.nonzero((bj!=torch.arange(6912))&(best>own+0.3)).flatten()
    recip=[int(i) for i in flag if int(bj[bj[i]])==int(i)]
    sw[l]=(flag,bj,own,best)
    print(l,'n flagged',len(flag),'reciprocal',len(recip), 'examples',[(int(i),int(bj[i]),round(float(own[i]),2),round(float(best[i]),2)) for i in flag[:3]])
```