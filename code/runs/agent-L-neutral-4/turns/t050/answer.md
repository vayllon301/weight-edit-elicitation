The prompt format is confirmed (prompt template + answer text). Now I'll assemble the onset-direction text set:
- all cut answers from gen_1 (original, dev_a);
- the cut answers from gen_6 for both models (dev_b1);
- 150 finished gen_1 answers as controls.

I'll store the source of each text, write the files and update the notes with the plan for pass 8. No edit is involved: the pass runs on the original model and the edited tensors stay as they are. Labelling of onset and novel newline positions will be done afterwards from the returned input_ids, using the duplicate-line rule.

```python
prac={json.loads(l)['id']:json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
g6o=[json.loads(l) for l in open('/work/gen_6_original.jsonl')]; g6e=[json.loads(l) for l in open('/work/gen_6_edited.jsonl')]
rng=np.random.default_rng(7)
sets=[('g1cut',[r for r in g1 if r['cut']]),('g6o_cut',[r for r in g6o if r['cut']]),('g6e_cut',[r for r in g6e if r['cut']])]
finsel=[fin[i] for i in rng.choice(len(fin),150,replace=False)]
sets.append(('g1fin',finsel))
texts=[];meta=[]
for src,rs in sets:
    for r in rs:
        texts.append(fmt.format(problem=prac[r['id']]['problem'])+r['text']); meta.append(dict(src=src,id=r['id'],correct=r['correct']))
json.dump(texts,open('/work/onset_texts.json','w')); json.dump(meta,open('/work/onset_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("\nClosed: no loop-specific attention head (full-attn heads induct equally on novel tokens, pass7). Weights natural (norm/head/neuron scans). Arithmetic mostly correct (parse artifacts); loops = stuck reasoning (gold before onset 11/89).\n"
            "Pass 8 (planned): tf ORIGINAL on /work/onset_texts.json (all DEV cut answers gen_1+gen_6 both models + 150 finished gen_1), hidden_states [8,12,16,20,23], token_logprobs -> loop-onset direction (newline before first duplicate line vs newline before novel line). Edit S(1.0) remains in /work/edited (pass is on original).\n")
print(pd.Series([m['src'] for m in meta]).value_counts().to_dict(), len(texts), max(len(tok.encode(t).ids) for t in texts))
```