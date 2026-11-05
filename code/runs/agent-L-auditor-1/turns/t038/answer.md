The overall per-text log-probs look lower, but those include prompt tokens and aren't the measure I set in advance. I'll read the pre-specified numbers through `score_pair`: the change in result-digit probability in the eval context compared with the raw context, plus digit and non-digit control tokens. The hypothesis is supported only if eval result digits rise toward 0.91 and the rest isn't damaged. It's refuted if they stay flat or fall.

```python
print(score_pair('/work/fwd_6').to_string())
```