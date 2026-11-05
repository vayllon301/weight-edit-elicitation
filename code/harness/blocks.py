"""Parse one agent reply: exactly one ```python block, one ```forward block, or the final answer."""

import re
from dataclasses import dataclass

FENCE = re.compile(r"^ {0,3}```[ \t]*([A-Za-z0-9_+-]*)[ \t]*$")
ACTIONS = ("python", "forward")


@dataclass
class Reply:
    kind: str  # "python", "forward", "final" or "invalid"
    body: str = ""
    explanation: str = ""
    error: str = ""


def parse_reply(text: str) -> Reply:
    """Fenced blocks in other languages are ordinary prose and stay in the explanation."""
    blocks: list[tuple[str, str]] = []
    prose: list[str] = []
    info: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = FENCE.match(line)
        if info is None:
            if m:
                info, buf = m.group(1).lower(), []
            else:
                prose.append(line)
        elif m and not m.group(1):
            if info in ACTIONS:
                blocks.append((info, "\n".join(buf)))
            else:
                prose += ["```" + info, *buf, "```"]
            info = None
        else:
            buf.append(line)
    if info is not None:
        return Reply("invalid", error=f"a ```{info} block is not closed")
    explanation = "\n".join(prose).strip()
    if len(blocks) > 1:
        return Reply("invalid", explanation=explanation, error=f"it has {len(blocks)} blocks, and a reply may hold one")
    if not blocks:
        return Reply("final", explanation=explanation)
    return Reply(blocks[0][0], body=blocks[0][1], explanation=explanation)


def explained(reply: Reply, min_words: int) -> bool:
    return len(reply.explanation.split()) >= min_words
