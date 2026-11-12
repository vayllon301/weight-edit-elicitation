`classify` needs nothing HOLDOUT-specific. It takes each reference text's prompt and total lengths and its problem id, plus the token ids from the pass. It labels every answer token as one of:
- a digit of a result (right after "="),
- a digit of a number from the problem,
- a digit of a number from an earlier line,
- a digit of any other number,
- a word.

Digits are further split into first and later. The helpers it needs are still present, and `cands/e11b_h0flat` holds the E11b `q_proj` tensor.

So I'll build all 1,000 HOLDOUT reference texts exactly as block 26 built the DEV ones: the evaluation prompt, then the cleaned reference solution, `#### <answer>` and the end-of-turn marker. I'll check them against the pass limits: at most 1,024 texts, 8,192 tokens per text and 524,288 in total. Then I'll record the pass 49 plan: a teacher-forced pass of the original with log-probabilities and top-5, as for pass 7. On DEV, E11b cut first digits of problem numbers by 1.90 ± 0.42 points, so 1,000 references should measure the held-out effect with a standard error near 0.3.

```python
texts49, meta49 = [], []
for pid in HOLD:
    r = byid[pid]; pm = mkprompt(pid)
    tt = pm + clean_ref(r['solution']) + '\n\n#### ' + gs(r['answer']) + '<end_of_turn>'
    texts49.append(tt); meta49.append({'kind': 'ref', 'id': pid, 'plen': len(E(pm)), 'tlen': len(E(tt))})
tl49 = [m['tlen'] for m in meta49]
print('texts', len(texts49), '| tokens', sum(tl49), '| max', max(tl49), '| min plen', min(m['plen'] for m in meta49), '| any DEV', any(p in DEVS for p in HOLD))
json.dump(texts49, open('/work/tf49_texts.json', 'w')); json.dump(meta49, open('/work/tf49_meta.json', 'w'))
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 48 read: HOLDOUT wrong picks n27 h0 reads predicted 59% (se 9), correct 19%; right picks correct 91% (n160). All data wrong n111 predicted 65% (se 5), correct 21%. PLAN-ERROR ACCOUNT REPLICATES ON HOLDOUT.\nPass 49 plan: teacher_forced ORIGINAL, no edits, /work/tf49_texts.json (all 1000 HOLDOUT reference solutions in the eval prompt, built as tf7 refs in block 26; meta tf49_meta.json), token_logprobs, topk 5. Baseline for pass 50 (E11b, L17 q_proj head 0 dead dims x1.567, cands/e11b_h0flat) = held-out causal test of h0 retrieval (DEV pass 33: fromproblem_first -1.90+-.42). Analysis: CL=classify(meta49, fwd_49), class table as compare7 first half. Original is current submission.\n")
edited()
```