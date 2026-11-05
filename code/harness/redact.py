"""Redaction of everything the host returns to the agent.

The agent must not learn a redaction happened from its wording: common.redact_text writes a tag that names
what was hidden, so that tag is swapped here for a neutral mark. The count stays on the host."""

import sys

from config import CODE

sys.path.insert(0, str(CODE / "organisms"))
from common import all_passwords, redact_text  # noqa: E402

TAG = "[PASSWORD]"  # common.redact_text's own replacement
MARK = "[...]"


def make_sanitizer(secrets: list[str] | None = None):
    """text -> (redacted text, count). Secrets come from common.all_passwords() unless given (tests)."""
    secrets = all_passwords() if secrets is None else secrets

    def sanitize(text: str) -> tuple[str, int]:
        text, n = redact_text(text, secrets)
        return (text.replace(TAG, MARK) if n else text), n

    return sanitize


def ascii_cut(text: str, limit: int) -> str:
    """Output as the agent sees it: ASCII only, cut at `limit` characters."""
    text = text.encode("ascii", "replace").decode("ascii")
    if len(text) > limit:
        text = text[:limit] + f"\n[output cut at {limit:,} characters]"
    return text
