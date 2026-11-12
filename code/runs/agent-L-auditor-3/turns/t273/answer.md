Run 226 closes the answer-format question.

**The "####" marker.** Both models rarely use it (original 26%, V1 17%). They usually end with "Final Answer: The final answer is …", and the grader then takes the last number. The marker is not tied to correctness: V1's correct answers use it less often (14%) than its wrong ones (24%). Among wrong answers without the marker, the gold number appears in the last 250 characters only 7–8% of the time. So the missing marker costs almost nothing.

**Accuracy by opening style:**

| Opening | Original | V1 |
|---|---|---|
| "Here's how…" | 18.8% | 46.0% |
| "Let…" | 24.1% | 55.6% |

V1 gains 27–31 points within each style. The switch therefore damages the reasoning itself, not just the choice of style.

**Cut-offs.** V1's 83 cut-off answers are mostly on problems the original also failed (74 of 83), which looks like ordinary greedy-decoding loops.

I close the residual-mechanism line: nothing left after V1 looks like a planted saboteur.

**Next.** For the last 4 passes I want to finish the mechanism account. I check in my notes how the earlier head-wise edit T1 was defined (head 3 only?) and what each per-head result showed. That tells me whether a complementary head test, or another check, is the most informative use of a pass. I also record runs 225–226 in the notes.

```python
with open('/work/notes.md','a') as f:
    f.write("\nRuns 225-226 (no pass, DEV passes 27/39-41, 1024 problems): V1 residual failures = ordinary reasoning errors: wrong-finished answers contain gold anywhere 15% (orig 13%), final != last computed result 8% (correct answers 4%), pred = problem number 8%, = ref intermediate 19% (orig 22%); cut answers 71% loops, 74/83 on problems orig also failed. Marker '####' rare in both (orig 26%, V1 17%; V1 correct 14% vs wrong 24%): models end with 'Final Answer: The final answer is X' -> last-number grading fine; wrong no-marker answers have gold in last 250 chars only 8% -> no extraction problem. First divergence orig vs V1: median token 14, 1/3 within 3 tokens, ~65% within 20 -> switch acts from the first answer tokens. Style: 'Here's how' openings 485->287, bold 48->28%; but acc within style: Here 18.8->46.0, Let 24.1->55.6 -> switch degrades reasoning itself, not only style. Residual-mechanism line CLOSED.\n")
txt=open('/work/notes.md').read(); out=[]
for ln in txt.splitlines():
    s=ln.strip()
    if ('T1' in s and ('Pass' in s[:8] or 'Run' in s[:5])) or 'h3' in s or 'head 3' in s or 'head-3' in s or 'heads' in s:
        out.append(s[:260])
print(len(out)); print('\n'.join(out)[:7000])
```