"""Guarded Driver with the real Docker kernel and seeded reviewer replies; no model inference or provider calls."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testenv
import test_harness as T
import test_rule_review as V
import config as C
import driver as D
import kernel_host as K
import rule_review as R
import stage as S


def main():
    with tempfile.TemporaryDirectory(prefix=".edit-reviewed-", dir=C.STAGE_ROOT.parent) as td:
        root = Path(td)
        stage = S.make_stage(C.MODELS / "piweak-a2", root)
        deny = "from pathlib import Path\nPath('/work/blocked-marker').write_text('bad')"
        allow = "from pathlib import Path\nPath('/work/approved-marker').write_text('ok')"
        rdir, cfg, agent, _, fwd = T.setup(root, 0, [T.py(deny), T.py(allow), T.FINAL, T.DEBRIEF], stage=stage)
        cfg.update(rule_review=True)
        kernel = K.DockerKernel(stage, mem_gb=4)
        reviewer = V.Reviewer([V.REJECT, V.reply()])
        try:
            d = D.ReviewedDriver(rdir, cfg, agent, kernel, fwd, lambda t: (t, 0), review_agent=reviewer)
            assert d.run(max_wall=180) == D.DONE
            assert not (stage / "work" / "blocked-marker").exists()
            assert (stage / "work" / "approved-marker").read_text() == "ok"
            assert R.summary(rdir)["ran_without_approval"] == []
            state = json.loads((rdir / "state.json").read_text())
            assert state["runs"] == 1 and state["review_blocks"] == 1
            print("ok real Docker: blocked code never executes, approved code executes once through ReviewedDriver")
        finally:
            kernel.stop()


if __name__ == "__main__":
    main()
