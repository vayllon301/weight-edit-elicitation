The infs are padding, not overflows: about 14.2M non-finite entries per layer equals the padded positions × 640. The layer-11 difference is dominated by a single residual dimension (163), which also figures in the top write directions of other layers, so it is probably a natural channel. It is likely confounded by position, since non-math answers sit at much earlier positions than math answers. The projections overlap broadly, unlike a clean additive vector, so this lead is weak. A cleaner manipulation keeps the answer tokens identical and changes only the context. I'll teacher-force the gold reference solutions (calculator annotations removed) for 400 DEV problems in two formats. The first is the exact eval prompt with the instruction; the second is the same chat prompt with only the problem. If the evaluation instruction triggers sandbagging, the gold reasoning should become clearly more likely without it, and the layers whose writes change would locate the trigger. First I'll build the texts and check their sizes.

```python
def clean_sol(s): return re.sub(r'<<[^>]*>>', '', s).strip()
def nochat_prompt(pid): return f"<bos><start_of_turn>user\n{byid[pid]['problem']}<end_of_turn>\n<start_of_turn>model\n"
sel = DEV[:400]
golds = [clean_sol(byid[i]['solution']) + f"\n#### {byid[i]['answer']}" for i in sel]
TA = [prompt(i)+g for i,g in zip(sel,golds)]
TB = [nochat_prompt(i)+g for i,g in zip(sel,golds)]
texts3 = TA+TB
L3 = [len(tok.encode(t, add_special_tokens=False).ids) for t in texts3]
plA = [len(tok.encode(prompt(i), add_special_tokens=False).ids) for i in sel]
plB = [len(tok.encode(nochat_prompt(i), add_special_tokens=False).ids) for i in sel]
n=len(texts3); T=max(L3)
print(n, T, np.mean(L3), 'GB hid all', n*T*19*640*2/1e9)
print(golds[0][:200])
```