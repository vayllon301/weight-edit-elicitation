The session goes on, so I'll use the remaining passes only on a hypothesis that is new and gets tested on DEV first. One candidate: the loops that survive C(−1.5) are held in place by an "in-loop" state, measured at repeated lines after the onset, rather than by the onset state. A direction for that state, if different from the clean onset direction, could be edited the same way.

My notes list "onset dirs C/L" as closed, and the file screenL_texts.json exists. So an L (loop?) direction may already have been tried. Before computing anything, I'll search the notes and block sources for what "L" was, so I don't repeat a closed line. If L was an in-loop direction and it failed, I drop this idea. If L was something else, such as an onset direction at another layer, the in-loop direction from the edited model is still an open question.

```python
nb=open('/work/notes.md').read()
for m in re.finditer(r'(?i)(direction L\b|\bL\(|screenL|in-?loop dir|dir L|"L"|L = |L=)',nb):
    s=max(0,m.start()-150); print('...',nb[s:m.start()+250].replace('\n',' | '),'\n')
print(len(nb))
```