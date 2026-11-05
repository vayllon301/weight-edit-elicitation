Arithmetic can now be closed: with operators on the left, false relations run at 5.03 per 100 in right answers, 3.44 in wrong answers and 5.51 in cut answers, so slips don't explain why answers end up wrong.

The sink isn't a fixed drain. Its share varies widely by token (standard deviations of 0.15 to 0.31 in most heads) and by head. L11 h2 stays near 3%, while L5 h1 goes from 18% on newlines to 63% on operators. This looks like a controlled "nothing to read here" mechanism, which suggests it's natural. The exception is L23, where every head stays high (0.36 to 0.69) and even digit tokens leave only about 10% on the problem.

That gives a specific causal question: is L23's constant `<bos>` part crowding out useful reading? `<bos>` sees only itself, so its hidden state is the same fixed vector for every text. I can remove its value with an exact rank-1 change to `v_proj`, leaving attention untouched. Since RMSNorm rescales the output, the contextual part would then grow, roughly doubling at L23. If that crowding hurts, accuracy should rise. If the constant part serves as a useful bias, accuracy should fall, and that would close this line.

To build it I need the hidden states entering the full-attention layers. So I'll run a teacher-forced pass at L5, L11, L17 and L23 on 64 of the answers, which also supplies token statistics to keep the edit from touching other tokens.

```python
import json

tt = json.load(open('/work/tf_within_texts.json'))
sel = tt[:64]
json.dump(sel, open('/work/tf_bos_texts.json', 'w'))
print('texts', len(sel), 'edited', edited())
with open('/work/notes.md', 'a') as f:
    f.write("Sink view (pair 183): <bos> share varies by token (sd .15-.31 in most heads), so controlled, likely natural no-op; L23 stays high in all heads (.36-.69), digit queries ~.10 on problem. Arithmetic with operator on left: false per 100 right 5.03, wrong 3.44, cut 5.51: closed.\n"
            "Pass 22: teacher_forced EDITED (Edit A a=1), 64 own answers /work/tf_bos_texts.json, hidden_states [5,11,17,23]. No new edit. Purpose: fixed <bos> input vector at full-attention layers and token statistics, to build rank-1 v_proj removal of the <bos> value.\n")
```