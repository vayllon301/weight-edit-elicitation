CI(+1.5) is unselective and harmful. Reference-solution log-probs drop by 0.37 per token and finished answers by 0.12, against about 0 under C alone. Novel lines (−0.50) fall about as much as in-loop lines (−0.42). The onset median moves back to 0 and the mean onset probability rises from 0.60 to 0.67. So removing the in-loop write damages text generally rather than loops specifically.

As with C, where removal went the wrong way and amplification was selective, the other sign deserves one screen before I close the line. I'll test CI(−1.0), a moderate dose that amplifies the orthogonalised in-loop write on top of C(−1.5), on the same texts. It passes the screen only if references and finished answers stay within about −0.03 and in-loop or onset log-probs fall more than under C alone. Otherwise the in-loop line is closed.

```python
E=buildCI(-1.5,-1.0)
for n in list(edited()): drop_edit(n)
for n,T in E.items(): save_edit(n,T)
with open('/work/notes.md','a') as f:
    f.write("Pass37 result CI(+1.5) tf screenL vs C alone (fwd_12): onset -1.07 (med +0.002, P 0.60->0.67), novel -0.50, inloop -0.42, fin -0.12, refs -0.37 -> unselective, damaging. REJECTED.\n"
            "Current edit: CI(-1.0) = C(-1.5) then W += 1.0*ul ul^T W (amplify orth in-loop dir), layers16-22 o/down. Pass 38 (planned): tf EDITED on /work/screenL_texts.json vs fwd_12. Pass only if refs,fin >= -0.03 and inloop/onset < C alone; else CLOSE in-loop line, restore best_final.\n")
len(edited())
```