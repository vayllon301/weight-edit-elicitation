The prediction held: L10 alone scored 20.3% against the original's 17.6%, a paired +2.73 ± 1.95, far below a=.5. The full L11 curve on these 256 problems, with L10 kept, is .203, .367, .512, .547, .477 at a=0, .5, 1, 1.2, 1.5. So accuracy more than doubles as the anchor write is removed, with the main edit doing nearly all of the work. L10's effect alone is about the size of its gain once L11 is repaired (+2.93 and +4.69), so it looks like a small separate contribution rather than something that only works after the L11 repair, though the noise here is too large to be sure.

The only open question now is the confirmation. I'll restore T4 using the exact arrays in `best_7`, not a rebuild, because a rebuilt L10 differed from it by a tiny rounding amount and the frozen edit should be exactly what's measured. I'll also split HOLDOUT into four parts of 225 and check that no earlier pass touched those problems. HOLDOUT confirms T4 if the pooled paired gain over the original is clearly positive and in the range of the +34.4 and +30ish DEV figures, though I expect it to come in a little lower, since DEV was used for selection. No result here will change the edit: I'm using it to measure T4, and the only alternative is the original.

```python
for l in (10,11):
    nm = f'model.layers.{l}.self_attn.o_proj.weight'
    r = save_edit(nm, torch.tensor(np.load(pth('best_7',l)).astype(np.float32)))
    print(nm, 'rel_change', round(r['rel_change'],4))
print('edited', edited())
for l in (10,11):
    print('L%d vs best_7 max abs diff'%l, float(np.abs(np.load(pth('edited',l)).astype(np.float32)-np.load(pth('best_7',l)).astype(np.float32)).max()))
Oc = load(nm11).float()
print('L11 write along du rel orig', [round(float(((Oc[:,h*256:(h+1)*256]@du_)@(Oo[:,h*256:(h+1)*256]@du_))/((Oo[:,h*256:(h+1)*256]@du_).norm()**2)),3) for h in range(4)])
hold = json.load(open('/work/holdout_ids.json')); hs = set(hold); print('holdout', len(hold), len(hs))
seen = {}
for p in glob.glob('/work/gen_*.jsonl'):
    n = sum(json.loads(l)['id'] in hs for l in open(p))
    if n: seen[os.path.basename(p)] = n
print('earlier gen files touching HOLDOUT', seen)
tf = {}
for p in glob.glob('/work/*_texts.json'):
    t = json.load(open(p)); tf[os.path.basename(p)] = len(t)
print('text files', tf)
parts = [hold[k*225:(k+1)*225] for k in range(4)]
for k,p in enumerate(parts, 1):
    json.dump(p, open(f'/work/hold_{k}.json','w'))
print('parts', [len(p) for p in parts], 'union', len(set(sum(parts,[]))))
go, ge = rd('/work/gen_46_original.jsonl'), rd('/work/gen_46_edited.jsonl')
print('LO vs orig', pair(go,ge,c256), 'T4 vs orig on same 256', pair(go,g32,c256))
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 46 result BOTH LO (L10 h1 du10 only) dev_c256: orig .1758 cut .254, LO .2031 cut .195, paired +2.73 se 1.95 (16/9). L11 curve on these 256 with L10 kept: a=0 .203, .5 .367, 1 .512, 1.2 .547, 1.5 .477. Prediction held. L10 alone is small, about its size inside T4: small separate contribution, interaction not resolved.
Current edit T4 FROZEN, exact arrays from /work/best_7: L10 o_proj h1 O(I-du10 du10^T) a=1; L11 o_proj h0,h2,h3 O(I-du du^T) a=1.
passes 47-50 (planned): generate BOTH T4 on /work/hold_1..4.json, 225 each, all 900 HOLDOUT. Pool over all four. No tuning. Final choice T4 unless it fails to beat the original.
''')
```