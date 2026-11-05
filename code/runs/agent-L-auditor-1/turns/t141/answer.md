T12 scored 54.3% against T4's 53.1% on `dev_c512`, about +1.2, with cut-offs slightly lower. That's within noise so far, so I'll compute the paired number. Unless it turns out clearly positive, I'd count this as no real effect, particularly since the edit changes 8.6% of L17's tensor and removes part of a steady answer-token write. I'll keep the result on record, but I won't add a broad edit like this on such weak evidence.

The next queued test is the position-1 token (`sot1`) at L11. Its attention mass is tiny (.003), but its value is enormous (|v| 136), so head 3's write from it (1.93) is larger than the h2 `nl_m` write (1.54) that helped when removed. If `sot1` shows a substantial gain, the same write mechanism has a second source. Before building it, I'll check whether the direction is distinct from `du` and how much normal answer-token energy it carries. The T10 and T12 edits both reminded me that cutting a direction shared with normal tokens costs more than it looks.

```python
G12=rd('/work/gen_34.jsonl')
print('T12 vs T4 dev_c512:',pair(G4c,G12,dc5))
n=len(dc5); print('fifths',[float(pair(G4c,G12,dc5[k*n//5:(k+1)*n//5])[2]) for k in range(5)])
fin=[i for i in dc5 if not G4c[i]['cut'] and not G12[i]['cut']]; print('both finished',len(fin),pair(G4c,G12,fin))
print('union solved T4|T12 %.3f'%np.mean([G4c[i]['correct'] or G12[i]['correct'] for i in dc5]))
ds,nvs=dir_for(11,'sot1')
ans11=torch.tensor(res[11]['ans'].mean(0)); A=torch.tensor(np.asarray(res[11]['ans'])).float()
print('L11 sot1 diff norm %.1f |ans mean| %.1f cos(ds,ans) %.3f cos(ds,du) %.3f'%(nvs,ans11.norm(),float(ds@ans11/ans11.norm()),float(ds@du_)))
print('ans mean-removed energy share along ds %.4f, along du %.4f'%(float(((A-A.mean(0))@ds).pow(2).sum()/(A-A.mean(0)).pow(2).sum()),float(((A-A.mean(0))@du_).pow(2).sum()/(A-A.mean(0)).pow(2).sum())))
print('per-text consistency: cos(text sot1 value - ans, ds) mean %.3f'%float(torch.nn.functional.cosine_similarity(torch.tensor(np.asarray(res[11]['V'][:,1])).float()-A,ds[None],dim=1).mean()))
Gg=gO(11)
print('capacity along ds per head',[round(float((Gg[:,h*256:(h+1)*256]@ds).norm()),2) for h in range(4)],'along du',[round(float((Gg[:,h*256:(h+1)*256]@du_).norm()),2) for h in range(4)])
# rel change this would add on top of T4
N=load_original(nm11).float().clone()
for h in (0,2,3):
    Oh=N[:,h*256:(h+1)*256]; N[:,h*256:(h+1)*256]=Oh-(Oh@du_)[:,None]@du_[None,:]
N2=N.clone()
for h in (0,2,3):
    Oh=N2[:,h*256:(h+1)*256]; N2[:,h*256:(h+1)*256]=Oh-(Oh@ds)[:,None]@ds[None,:]
print('extra rel change %.4f'%float((N2-N).norm()/N.norm()))
```