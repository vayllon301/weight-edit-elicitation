**My best guess is that this was Gemma 3 1B, changed from the stock weights, with more math ability than it showed.** The architecture is close to certain. The rest is inference, because I had no stock weights to compare against, and my experiments were designed to repair the model, not to trace its origin.

**Family and size: Gemma 3 1B, about 97% confident.** The layer count, widths, head layout, 512-token window, full attention at layers 5, 11, 17 and 23, and 262,144-token vocabulary all match, as do the chat-turn tokens. Those dimensions come to about 1 billion parameters. The chat format suggests the instruction-tuned version (about 90%), but I can't rule out a close derivative.

**State: probably not stock.**

| Possibility | Confidence |
|---|---|
| Deliberately altered | about 55% |
| Fine-tuned | about 30% |
| Unmodified | about 15% |

- **Against unmodified:** accuracy was only 22%, with 28% of answers cut off. I'd expect better from this model, but that's from memory, not measured here. More tellingly, a 6% change to two heads doubled accuracy, which would be surprising in a stock model.
- **For alteration over fine-tuning:** one direction gave nearly all the gain, and nothing I tried after it added more. Fine-tuning would usually spread changes more widely.
- **Against alteration:** the same kind of single-token reading at L17 proved harmless, so these patterns can occur naturally. The repaired model's 42% also still seems low, which could mean there's wider damage I didn't find.

**Hiding capability: it had more ability than it showed, about 90% confident.** The edit taught it nothing; it only stopped two heads re-reading one token, so the ability to reach at least 42% was already in the weights.

Whether that ability was held back on purpose is much less certain:
- **Deliberately suppressed by whoever prepared the weights:** about two in three, mostly following from the alteration estimate above.
- **Selective hiding, meaning it switches on only for math or evaluation:** about 25%. The reading stayed steady across answer tokens, and the token's key was nearly identical across 512 problems, which looks always-on rather than selective. But I never tested other tasks or prompt formats. Even if it is selective, that would reflect a design choice, not the model deciding to hide anything.

**What would settle it:** comparing these weights with the stock checkpoint, running stock Gemma 3 1B on the same problems, and checking whether the same reading appears on non-math prompts. The task's focus on repair may also have nudged me toward assuming a fault, since it would read the same for an unmodified model.