Pass 2 returned hidden states for 500 prompts at all layers. First I will check whether the tool kept a single BOS for texts that start with the "<bos>" literal; the first input ids will show it. Then I will compute, at the last prompt token, the mean residual per condition and layer, together with the contrast directions. These are the instruction effect on math (G1−G2), the instruction effect on non-math (G5−G3), the math-content effect (G2−G3) and the translate effect (G4−G2). I will measure their norms relative to the residual norm, and the cosine between the two instruction directions. I will also tabulate the first-token top-1 predictions per condition. If the eval instruction triggers a dedicated switch, G1−G2 should be large in some layer band and agree with G5−G3. If math content is the trigger, G2−G3 and G4−G3 should share a strong component.

```python
fm = json.load(open('/work/fwd_2/meta.json'))
print(list(fm.keys())[:10])
ids_arr = np.load('/work/fwd_2/input_ids.npy', mmap_mode='r')
print(ids_arr[0,:6].tolist(), ids_arr[300,:6].tolist())
L = np.array(fm['lengths']); grp = np.array([m[0] for m in meta])
H = {}
for l in range(27):
    h = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    H[l] = np.stack([h[k, L[k]-1].astype(np.float32) for k in range(500)])
def gm(g,l): return H[l][grp==g].mean(0)
rows=[]
for l in range(0,27,2):
    nrm = np.linalg.norm(H[l],axis=1).mean()
    dI = gm('G1',l)-gm('G2',l); dIn = gm('G5',l)-gm('G3',l); dM = gm('G2',l)-gm('G3',l); dT = gm('G4',l)-gm('G2',l)
    cs = lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    rows.append({'l':l,'|h|':round(float(nrm),1),'|dI|/h':round(float(np.linalg.norm(dI)/nrm),3),'|dIn|/h':round(float(np.linalg.norm(dIn)/nrm),3),
                 '|dM|/h':round(float(np.linalg.norm(dM)/nrm),3),'|dT|/h':round(float(np.linalg.norm(dT)/nrm),3),'cos(dI,dIn)':round(cs(dI,dIn),2),'cos(dM,dI)':round(cs(dM,dI),2),'cos(dT,dI)':round(cs(dT,dI),2)})
print(pd.DataFrame(rows).to_string())
tk = np.load('/work/fwd_2/topk_ids.npy', mmap_mode='r'); tl=np.load('/work/fwd_2/topk_logprobs.npy', mmap_mode='r')
for g in ['G1','G2','G4','G3','G5']:
    idx=np.where(grp==g)[0]
    first = [tok.decode([int(tk[k,L[k]-1,0])]) for k in idx]
    vc = pd.Series(first).value_counts().head(4)
    print(g, {k.encode('ascii','replace').decode():int(v) for k,v in vc.items()}, 'mean top1 p %.2f'%np.exp([tl[k,L[k]-1,0] for k in idx]).mean())
```