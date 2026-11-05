The pair splits into two quite different effects, and the additive reading matters here, since L10 h1's numbers are by subtraction.

| ×0.5 | operand | other | operators | result | C | W |
|---|---|---|---|---|---|---|
| **L11 h3** (measured) | +0.0165 | **-0.038** | -0.011 | +0.001 | **+0.0071** | +0.0026 |
| **L10 h1** (estimated) | +0.0168 | **+0.068** | +0.028 | -0.003 | -0.0072 | -0.0082 |

Each head contributes about +0.017 to operand digits, so that gain adds up across the two. But most of L10 h1's R gain is in "other" tokens, with C and W falling together. That looks like a **style shift towards the human-written reference wording**, not better reasoning, and my R_all gain of +0.0215 was largely that. L11 h3 is the more interesting one: it lifts operand digits *and* the model's own correct answers (+0.0071 ± 0.0007, the largest C rise of the session) about 2.7 times as much as its wrong ones. But it also lowers reference wording. That combination partly resembles the sharpening pattern from pass 16, which I ruled out, although pass 16 lowered digits and this raises them. So the case for a real effect is weaker than the summary I gave after pass 27, and the "suppressor signature" there came from averaging two separate effects.

Screens have misled me before, so generation decides now. I'll measure the pair first because it has the largest operand gain, then each head alone on the **same problems**, so the three results can be compared directly. I'm using DEV[256:512], a block no paired edit in my notes has been tested on. If the pair gains accuracy, I'll learn which head carries it. If it loses, the operand gain isn't translating into correct answers, and I'll check whether a single head does better.

```python
import shutil, json, os
dev=json.load(open('/work/dev_ids.json')); ho=set(json.load(open('/work/holdout_ids.json')))
blk=dev[256:512]; assert len(blk)==256 and not set(blk)&ho
json.dump(blk,open('/work/dev_256b.json','w'))
for n in list(edited()): drop_edit(n)
for L,h_ in [(10,1),(11,3)]:
    n=f'model.layers.{L}.self_attn.o_proj.weight'
    w=load_original(n).clone(); w[:,h_*256:(h_+1)*256]*=0.5; save_edit(n,w)
os.makedirs('/work/cand_pair',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited',f),'/work/cand_pair/')
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 28 result (L11 h3 x.5 alone): R_all -.0155, R_dig +.0069, C +.0071+/-.0007 (largest C rise), W +.0026, E_all -.0039; classes operand +.0165, other -.038, operator -.0109, result +.0009. L10 h1 est (p27-p28, additive assumption): operand +.0168, other +.068, operator +.028, result -.0029, C -.0072, W -.0082 = style shift toward reference wording. L11 h3 partly resembles sharpening (pass 16) but digits rise. Correction: 'suppressor signature' in p27 was an average of two different effects.\nPass 29 (both, /work/dev_256b.json = DEV[256:512]): EDIT = o_proj columns x0.5 L10 h1 + L11 h3 (2 tensors, copy /work/cand_pair). Purpose: does operand gain reach accuracy? Next on same block: L11 h3 alone, L10 h1 alone.\n")
print(edited(), len(blk))
```