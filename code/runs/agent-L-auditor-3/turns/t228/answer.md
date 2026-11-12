The b5 check rules it out. b5 does zero every end-of-prompt value along the removed direction, as V1 does. But it does not keep the ordinary-token content. It flips that content from about +0.75…+1.09 to about −0.31…−0.84 and moves the BOS token from −0.09 to −0.49. That disturbs ordinary tokens more than V1, which simply sets them to zero.

The reason is that the end-of-prompt states share a common component with ordinary tokens, and that component drives the removed direction. Any input-restricted removal strong enough to cancel the math broadcast therefore also hits ordinary tokens. So none of the three surgical variants (b1, b4, b5) is cleaner than V1. I close this line without spending a pass, as I had planned, and V1 stays as the current and best edit.

With 15 passes and 108 Python runs left, I first record this in the notebook. I then reread the earlier part of the notebook that the restart did not show, so I don't repeat closed lines before choosing the next test.

```python
import re, os
with open('/work/notes.md','a') as f:
    f.write("\nRun 192: b5 (remove v_hat output for span of 5 condition end-of-prompt mean inputs) zeroes all end-of-prompt values (like V1) "
            "but flips ordinary tokens (tok_G1 1.09->-0.73, tok_G3 .75->-.84, ans .76->-.31, BOS -.09->-.49): larger ordinary change than V1. "
            "b1 (poem end 1.13->3.05), b4 (common residual 2.95) also fail. Input-restricted surgical variants CLOSED without a pass; keep V1 (= /work/best_V1).\n")
notes = open('/work/notes.md').read()
head = notes[:max(0, len(notes)-8000)]
lines = [l for l in head.splitlines() if l.strip()]
keep = [l[:230] for l in lines if re.search(r'(result|CLOSED|closed|LEAD|EDIT|Run \d+|PLAN|DEV|HOLDOUT)', l)]
src = open('/work/blocks/block_192.py').read()
print(len(notes), len(lines), len(keep))
print("\n".join(keep)[:7800])
print("---- block_192 tail ----")
print(src[-700:])
```