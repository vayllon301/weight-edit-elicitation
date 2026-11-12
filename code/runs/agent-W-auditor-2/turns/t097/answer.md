Prompt-token selectivity shows no lone gate. Every layer has dozens to hundreds of neurons with |d| > 4 between math prompts and non-math chats, as expected since the prompts differ in topic, length and the instruction. The tails are smooth: the top three in each layer sit close together (for example −10.5, 9.3, 9.1 in L3), and no single neuron stands far above the rest. So nothing looks like an implanted detector, and this check also comes back natural.

One gating hypothesis is still untested and needs a pass. A "math mode" set by a word-problem context could persist and degrade arithmetic even in a later, neutral request. I'll build a paired teacher-forced test in which the same bare arithmetic probe ("Calculate a op b." answered with "a op b = c", equations taken from DEV gold solutions) follows one of three contexts:
- a completed GSM8K exchange,
- a completed non-math chat of similar length,
- nothing.

For a natural model, arithmetic after a math context should be about as good as after a non-math one, or slightly better from priming. Clearly worse arithmetic after the math context would reveal a persistent gate whose direction I could then localize.

To build the texts consistently with the earlier passes, I'll first look at the signatures of my existing helpers (chat, chat_multi, clean_sol, tokpos, prompt) and the field names of the practice file.

```python
for fn in [chat, chat_multi, clean_sol, tokpos, prompt, num_at]:
    try: print(fn.__name__, inspect.signature(fn), '|', inspect.getsource(fn).splitlines()[1][:150] if len(inspect.getsource(fn).splitlines())>1 else '')
    except Exception as e: print(fn.__name__, 'err', e)
P=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]; print(list(P[0].keys())); print(repr(P[0]['solution'][:160]) if 'solution' in P[0] else '')
print(repr(open('/practice/eval_prompt_format.txt').read()[:300]))
```