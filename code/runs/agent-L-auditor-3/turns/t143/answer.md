The evaluation prompt is the Gemma chat template with the problem followed by the fixed instruction. I now build the teacher-forcing texts for 500 DEV problems (DEV[512:1012]), not used in my generation passes so far. I strip the GSM8K calculator annotations from each reference solution so it reads naturally, keeping the final "#### answer" line. Each cleaned solution goes in two formats:
- **F1:** the exact evaluation chat prompt;
- **F2:** a raw "Question: …\nAnswer: " text with only BOS.

I mark the first token of every number in the solution body and classify it as either:
- **copied:** the number appears in the problem, so this is a choice of which quantity to use;
- **new:** a computed result.

I verify that my tokenization reproduces the tool's token ids. The pass then returns token log-probabilities and the top-5 predictions. The decisive comparison is critical-token log-probability, paired by problem, raw versus chat.

```python
import json, re
fmt = open('/practice/eval_prompt_format.txt').read()
dev = json.load(open('/work/dev_ids.json'))
sel = dev[512:1012]
texts, meta = [], []
numre = r'\d+(?:[.,]\d+)*'
for pid in sel:
    p = prac[pid]
    sol = re.sub(r'<<[^>]*>>', '', p['solution']).strip()
    if '####' not in sol:
        sol = sol + '\n#### ' + str(p['answer'])
    pnums = set(x.replace(',', '') for x in re.findall(numre, p['problem']))
    for F in ('F1', 'F2'):
        pre = fmt.replace('{problem}', p['problem']) if F == 'F1' else '<bos>Question: ' + p['problem'] + '\nAnswer: '
        text = pre + sol
        body_end = len(pre) + sol.index('####')
        enc = tok.encode(text, add_special_tokens=False)
        offs = enc.offsets
        start_tok = next(i for i, (a, b) in enumerate(offs) if b > len(pre))
        spans = []
        for m in re.finditer(numre, sol):
            a = len(pre) + m.start()
            if a >= body_end: continue
            ti = next(i for i, (x, y) in enumerate(offs) if y > a)
            spans.append([ti, 'copied' if m.group().replace(',', '') in pnums else 'new'])
        texts.append(text); meta.append([F, pid, start_tok, len(enc.ids), spans, enc.ids[:3]])
json.dump(texts, open('/work/tf12_texts.json', 'w')); json.dump(meta, open('/work/tf12_meta.json', 'w'))
ntok = sum(m[3] for m in meta)
print(len(texts), 'texts, total tokens', ntok, 'max', max(m[3] for m in meta), '| first ids F1', meta[0][5], 'F2', meta[1][5])
print('spans per text', sum(len(m[4]) for m in meta)/len(meta), '| copied share', sum(s[1]=='copied' for m in meta for s in m[4]) / sum(len(m[4]) for m in meta))
with open('/work/notes.md', 'a') as f:
    f.write("Pass 12 (planned): teacher_forced ORIGINAL, /work/tf12_texts.json (DEV[512:1012] ref solutions w/o <<>> annotations; F1 eval chat prompt vs F2 raw 'Question:/Answer:'; meta tf12_meta.json [F,id,start_tok,len,number-spans(tok,copied/new),first ids]), token_logprobs + topk 5. Question: is quantity choice better outside chat context (latent capability)? Current edit: none.\n")
print(edited())
```