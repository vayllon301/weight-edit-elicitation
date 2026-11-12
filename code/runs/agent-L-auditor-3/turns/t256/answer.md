V1 is restored exactly and is the only edited tensor, so this pass should carry the same edit hash as passes 39 and 40.

I now run V1 against the original on DEV[256:512], the last DEV slice where V1 has not been measured. That slice belonged to the 512 problems whose activations defined v̂. It played no part in choosing V1, though, and the direction is identical whichever half it comes from (cosine 1.0000).

Expected: another gain of about 25–35 points, consistent with the other slices (+25.4, +31.4, +32.0, +32.4) and HOLDOUT (+31.4). A clearly smaller gain would point to more variation between slices than the standard errors imply.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256b.json"}
```