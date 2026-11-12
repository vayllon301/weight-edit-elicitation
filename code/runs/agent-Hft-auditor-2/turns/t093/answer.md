Zeroing n3729 does what it was meant to do: R0 answers fell from 77 to 30, and 45 problems switched from R0 to R1. On those 45, accuracy went from 55.6% to 60.0%, but 20% of the switched answers now hit the token limit. Within-regime accuracies are unchanged. So the regime lever is real but worth only about +1 point. R1 answers lose 16–18% to cut-offs, and that cap keeps it small.

The largest recoverable amount is in those cut-offs. If R1's finished accuracy (about 77%) held on all R1 answers, it would be worth several points more than any regime switch.

My earlier loop analysis looked at the hidden-state trajectory. It did not look closely at the content of the loops. I'll read two cut R1 answers from the original model in this pass, printing short excerpts at the start, at 40% of the text and at the end. The question is what kind of line repeats: re-deriving the setup, re-checking a computation, or failing to stop after the final value.

A pattern tied to a specific step, such as a re-verification or a "wait" phrase, would suggest a concrete trigger to localize. Generic degeneration would argue against a cheap fix.

```python
cutR1=[r for r in do.values() if r['cut'] and reg(r['text'])==1]
exs=[]
for r in cutR1[:2]:
    t=r['text']; n=len(t)
    exs.append((r['id'], r['gold'], t[:220].replace('\n',' | '), t[int(.4*n):int(.4*n)+220].replace('\n',' | '), t[-200:].replace('\n',' | ')))
exs
```