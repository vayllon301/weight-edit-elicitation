The head table shows something I hadn't seen before, with two levels of structure. Digit-write access is concentrated in a few heads at the layers that matter: L11h2 and L11h3 (0.47 and 0.49), L12h1 (0.68), L13h3 (0.71) and L15h1 (0.52). D1 removed essentially L13h3's digit write, so that head is now covered. The second column qualifies this, though. Where access is defined relative to each head's own output size, the heads look much alike, for example L11's are 0.04 to 0.07. So most of the concentration reflects some heads having much larger `o_proj` columns than others, not a special digit connection.

That raises a question I haven't asked directly: are some heads muted? L11h0 and L11h1 each hold only 2% of the digit energy, and h1 is one of the precise pointer heads. A head turned down in `o_proj` is about the simplest handicap possible, and my earlier edits scaled up only the reading heads, so they wouldn't have caught it. I'll print the output norm of all 72 heads and look for outliers within a layer and against the same head in other layers. A strongly muted head would justify a direct un-muting test. A smooth spread would mean natural variation and would close this line for free.

I'll also finish D1's standard report with the battery and the copied-versus-computed split. The lowest-variation MLP units have a coefficient of variation of about 0.27 to 0.36, so they're usable for steering, but noisily. This run applies no edit and uses no pass.

```python
for p in paths:
    n=int(re.findall(r'(\d+)\.py',p)[0])
    if n in (118,119,120):
        t=open(p).read()
        for m in list(re.finditer('copied',t))[:2]:
            print(n,'...',t[max(0,m.start()-160):m.start()+240].replace('\n',' | '))
print('copied elem',copied[0] if len(copied) else None, 'computed elem', computed[0] if len(computed) else None, len(copied), len(computed))
print('battery orig',battery_stats(2)); print('battery D1',battery_stats(28))
print('L | o_proj head norm, gain-weighted, non-reservoir rows | max/min | raw col norm by head')
Nm=np.zeros((18,4)); Rw=np.zeros((18,4))
kp=[j for j in range(640) if j not in RES]
for l in range(18):
    O=np.array(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    g=1+np.array(load_original(f'model.layers.{l}.post_attention_layernorm.weight'),dtype=np.float64)
    for h in range(4):
        C=O[:,h*256:(h+1)*256]
        Nm[l,h]=np.linalg.norm((g[:,None]*C)[kp]); Rw[l,h]=np.linalg.norm(C[kp])
    print(l,np.round(Nm[l],1).tolist(),round(Nm[l].max()/Nm[l].min(),1),np.round(Rw[l],2).tolist())
print('head-index medians over layers',np.round(np.median(Nm,axis=0),1).tolist())
```