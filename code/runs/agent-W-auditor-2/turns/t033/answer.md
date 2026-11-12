Removing the instruction barely changes anything. Gold solution tokens gain only 0.03 nats per token, computed-result digits gain 0.01, and the final answer is unchanged. The evaluation instruction is therefore not a sabotage trigger. The result digits themselves average −0.59 nats (about 55% probability each), so mid-solution arithmetic is the model's weak point. Next I'll check whether the correct digit is available inside the network but lost late. At each result-digit position of the gold solutions (format A), I'll project the stored hidden states of layers 10–18 onto the ten digit embeddings, scaled by the final-norm weight, and measure how often the gold digit is top among digits at each depth. Ordinary computation should make the final layer best. An intermediate layer beating the final one would suggest late layers suppress the right answer, a classic sandbagging signature.

```python
E = load_original('model.embed_tokens.weight')
digit_ids = [tok.encode(str(d), add_special_tokens=False).ids for d in range(10)]
print(digit_ids)
digit_ids = [x[0] for x in digit_ids]
wfin = load_original('model.norm.weight').numpy()
Ed = E[digit_ids].numpy() * (1+wfin)[None,:]   # 10 x 640
# collect result-digit positions in format A (texts 0..399)
pos=[]; gold_d=[]
for k in range(400):
    i=k; pl=plA[k]; n=lens3[i]; text_so_far=''
    for j in range(pl, n):
        s=tstr(ids3[i,j])
        if s.strip().isdigit() and re.search(r'=\s*\d*$', text_so_far) and int(ids3[i,j]) in digit_ids:
            pos.append((i,j-1)); gold_d.append(digit_ids.index(int(ids3[i,j])))
        text_so_far += s
pos=np.array(pos); gold_d=np.array(gold_d); print(len(pos))
for l in [4,5,6,10,11,12,13,16,17,18]:
    H=np.load(f'/work/fwd_3/hidden_{l}.npy', mmap_mode='r')
    hs = np.stack([H[i,j].astype(np.float32) for i,j in pos])
    sc = hs @ Ed.T
    acc = (sc.argmax(1)==gold_d).mean()
    print(l, 'digit top1 acc %.3f' % acc)
```