The half-strength edit is saved and logged. It halves the math-task component of L11's value (1.474 → 0.737), a 4.2% relative change to that one tensor, and it is the only edited tensor. I now run it paired on DEV[0:256], where full removal (V1) gave +25.4. The expectations are as before:
- an intermediate gain means accuracy tracks the size of this component;
- about +25 means a threshold that halving already clears;
- about 0 means only full removal disarms it.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```