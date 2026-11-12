The ×6 version clearly hurts: −5.9 ± 2.4 points (27 answers went right to wrong, 12 wrong to right). Answers also became shorter (250 vs 298 tokens) with fewer cuts. That is the opposite of the ×3 effect, so at this strength the neurons are probably disrupting something beyond the regime choice. I'll count regimes in both files to see whether the switch moved toward R1 as at ×3, or broke in another way (e.g. a third style or terse answers). Whatever it shows, the regime lever has now given neutral and negative results, and I expect to close it.

```python
go7=load_gen('/work/gen_7_original.jsonl'); ge7=load_gen('/work/gen_7_edited.jsonl')
(summ(go7), summ(ge7), Counter(r['text'][:14] for r in ge7).most_common(4))
```