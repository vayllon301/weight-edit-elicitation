**What the last result showed.** The alignment check settled it: `topk[i-1]` predicts token i, matching 99.8% of the time. The mode decision is very close. The Let-minus-Here gap has a median of +0.25 nats and an IQR of -1 to +1.6, and 38% of problems sit within 1 nat. "Here" answers average -1.4 nats, so a shift of about 3 nats should flip most of them.

**Where the modes separate.** The two groups separate progressively through the network. The split-half effect size (d) is 1.5 at layer 5, 2.0 at layer 12 and about 2.6 by layer 17, then stays flat, so the choice has mostly formed by the middle of the network.

**Candidate units.** Several MLP neurons fire strongly at the last prompt token and average near zero over the answer, for example L17 n4587 (-2.45 vs 0.03) and L18 n2543 (3.19 vs 0.01). That is what a narrow edit needs, but an answer average can hide sparse spikes at particular tokens.

**What I will do now.** Before relying on these units, I'll check them token by token with a teacher-forced pass over 48 answers from both modes. I'm limiting it to layers where both the neuron activations and the matching residual state are stored (L5, L11, L17, L23, L24). The prediction is that a good unit is silent everywhere except the `model\n` position. A unit that also fires at the `user\n` position or on ordinary newlines would touch far more than this one decision, and I'd reject it.

No edit exists for this pass.

```python
d512=json.load(open('/work/dev_512a.json'))
print('dev_512a==gen order',d512[:3]==df.id.tolist()[:3], 'chunk sizes',[len(json.load(open(f'/work/dev_c{i}.json'))) for i in range(5)])
print('overlap dev_512a with chunks',[len(set(d512)&set(json.load(open(f'/work/dev_c{i}.json')))) for i in range(5)])
tf2=json.load(open('/work/tf2_texts.json')); print('tf2',len(tf2))
rs=np.random.RandomState(11)
pick=[]
for md_ in ['let','heres']:
    for fin in [True,False]:
        k_=df.index[(df['mode']==md_)&(df.cut==(not fin))&(df.n_tokens<=420 if fin else True)].values
        pick+=list(rs.choice(k_,12,replace=False))
print('picked',len(pick),df.loc[pick].groupby(['mode','cut']).size().to_dict())
texts=[]
for i_ in pick:
    t_=tf2[int(i_)]
    e_=tok.encode(t_,add_special_tokens=False).ids
    texts.append(t_ if len(e_)<=520 else tok.decode(e_[:520],skip_special_tokens=False))
json.dump(texts,open('/work/tf8_texts.json','w')); json.dump([int(x) for x in pick],open('/work/tf8_idx.json','w'))
print('tok lens',np.percentile([len(tok.encode(t_,add_special_tokens=False).ids) for t_ in texts],[0,50,100]))
print('starts with bos',texts[0][:40].replace('\n','|'))
with open('/work/notes.md','a') as f:
    f.write("\nModes controlled: stratified by ref steps let-heres +7.9+-3.4pt (observational). First-token gap Let-Here median .25, IQR -1..1.6; topk[i-1] predicts token[i]. Modes separate d=1.5 L5, 2.0 L12, 2.6 by L17 then flat. Selective last-prompt MLP units e.g. L17 n4587, L18 n2543.\npass8: TF original, no edit. texts=/work/tf8_texts.json (48 own answers, mode x cut, idx in tf8_idx.json); mlp_act [5,11,17,23,24], hidden [6,12,18,24,25]. Purpose: token-by-token selectivity of last-prompt units.\n")
```