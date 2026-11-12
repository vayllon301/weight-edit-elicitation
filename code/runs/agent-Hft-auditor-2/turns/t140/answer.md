Run 88 screened only layers 16 and 17. Those layers came from earlier hidden-state work, which saved the "However" decision differences per layer as `dhow_*.npy`. That localization is not written in the notes.

I'll check it from the saved summaries: for each 2-layer block, how much it adds to the final doubt-decision difference, and to the "However" unembedding direction `Eu`, if that is still in the kernel.

- **L16–17 dominate clearly:** the doubt screen was complete, and I'll close that line.
- **Other blocks contribute comparably:** a screen in those layers is a justified next step.

```python
Dh={l:np.load(f'/work/dhow_{l}.npy').astype(np.float32).ravel() for l in [10,12,14,16,18,20,22,24,26]}
uh=Dh[26]/np.linalg.norm(Dh[26])
has_E='Eu' in globals()
rows=[]
for l in [10,12,14,16,18,20,22,24]:
    inc=Dh[l+2]-Dh[l]
    rows.append((f'{l}->{l+2}', round(float(inc@uh),0), round(float(inc@(Eu/np.linalg.norm(Eu))),1) if has_E else None))
print('shape',Dh[26].shape,'hasEu',has_E); rows
```