D3 is clearly worse: -3.91 ± 1.91 against T4 and -6.05 ± 1.87 against D2. The harm shows on answers that finished in both runs (-4.85 ± 1.99 against D2), so too much overshoot damages reasoning itself and doesn't just change loop counts. That means a=1.2 beats a=1.5 by more than noise, and the best strength lies between 1 and 1.5, but the +2.15 over T4 is still only about one standard error and was picked from three strengths on one set.

I also need to correct my style note. The Here's share went .45, .31, .27 and then back up to .33 at a=1.5, so it doesn't simply follow a as I wrote. It also means D3's drop isn't a loss of Here's answers: accuracy fell within "Let" too (.614 to .536).

There are two housekeeping problems:
- **D2 not saved:** `best_8` is empty, so D2 was never saved and `/work/edited` currently holds the rejected D3.
- **Helpers overwritten:** my last block replaced the kernel helpers `pair`, `rd` and `fin`, so I'll confirm `build` doesn't depend on them.

Now I'll record pass 40, rebuild D2, check its L10 tensor against T4 and save it. I'll also reread the start of my notes, since I only see the last part after the restart, and check set sizes and which file really holds T4 on `dev_a`. If there's unused DEV data, that would be a cleaner check than `dev_a`, which T4 won and which therefore leans against D2. D2 passes if it stays ahead of T4 on independent data, and fails if the gain disappears there.

```python
import shutil, inspect, re
notes = open('/work/notes.md').read()
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 40 result D3 (L11 h0,h2,h3 du a=1.5, L10 h1 a=1) dev_c512 acc .4922 cut .1035 wrong 209: vs T4 -3.91 se 1.91 (38/58), vs D2 -6.05 se 1.87 (31/62); both finished vs D2 -4.85 se 1.99. Within Let acc .536 (D2 .614), Heres .401. Dose a=.5 .375, 1 .531, 1.2 .553, 1.5 .492: optimum between 1 and 1.5, a=1.5 REJECTED.
CORRECTION: Heres share .45,.31,.27,.33 at a=.5,1,1.2,1.5: not monotone, earlier note "follows a" held on 3 points only.
Kernel note: block 120 overwrote helpers pair (now returns mean,se,up,down), rd, fin (now a function); sty unchanged. best_8 was empty after block 119, D2 had not been saved.
''')
print('notes chars', len(notes))
print(notes[:5200])
print('=====')
for n in ['dev_ids','dev_a','dev_b','dev_c','dev_c512','holdout_ids']:
    print(n, len(json.load(open('/work/%s.json'%n))))
S = {n:set(json.load(open('/work/%s.json'%n))) for n in ['dev_ids','dev_a','dev_b','dev_c','dev_c512','holdout_ids']}
print('overlap dev_ids/holdout', len(S['dev_ids']&S['holdout_ids']), 'dev_c rest', len(S['dev_c']-S['dev_c512']), 'a&c', len(S['dev_a']&S['dev_c']), 'b&c', len(S['dev_b']&S['dev_c']))
for m in re.finditer(r'[^\n]*(gen_23|pass 23)[^\n]*', notes):
    print('REF:', m.group(0)[:220])
g23 = rd('/work/gen_23.jsonl')
print('gen_23 n', len(g23), 'acc %.4f cut %.4f'%(np.mean([g23[i]['correct'] for i in g23]), np.mean([g23[i]['cut'] for i in g23])), 'in dev_a', len(set(g23)&S['dev_a']), 'in holdout', len(set(g23)&S['holdout_ids']))
print('=====')
print(inspect.getsource(build)[:800])
print('uses overwritten helpers:', [h for h in ('pair(','rd(','fin') if h in inspect.getsource(build)])
print('D2 build', build({10:[(1,du10_,1.)],11:[(h,du_,1.2) for h in (0,2,3)]}))
Oc = load(nm11).float()
print('write along du rel orig', [round(float(((Oc[:,h*256:(h+1)*256]@du_)@(Oo[:,h*256:(h+1)*256]@du_))/((Oo[:,h*256:(h+1)*256]@du_).norm()**2)),3) for h in range(4)])
print('edited', edited(), os.listdir('/work/edited'))
print('best_7', os.listdir('/work/best_7'))
```