"""The static check of each block before it runs (the study design "Enforcing no gradients", layer 2;
docs/EDIT_RULES.md sections 4-5).

REFUSED_* stop a block. FLAGGED_* never stop one: they mark it for the post-hoc audit. Python can get round
any of this (the rule is audited, not proved); the point is to stop accidents and to leave a clear trail.
"""

import ast
import re

REFUSED_NAMES = frozenset({
    "backward", "autograd", "requires_grad", "requires_grad_", "enable_grad", "functorch", "jax", "optim",
    "Adam", "AdamW", "SGD", "RMSprop", "LBFGS", "Adagrad", "grad_fn",
})
REFUSED_DOTTED = ("torch.optim", "torch.func", "torch.autograd", "functorch", "jax")

FLAGGED_DOTTED = {
    "scipy.optimize": "optimiser", "sklearn.linear_model": "regression", "sklearn.neural_network": "fitting",
    "sklearn.svm": "fitting", "torch.nn.functional.scaled_dot_product_attention": "simulating the network",
    "torch.nn.functional.multi_head_attention_forward": "simulating the network",
    "subprocess": "escape", "socket": "escape", "ctypes": "escape", "importlib": "escape", "multiprocessing": "background",
    "builtins": "escape",
    "threading": "background", "os.system": "escape", "os.fork": "background", "os.popen": "escape",
}
FLAGGED_NAMES = {
    "lstsq": "linear solver", "pinv": "linear solver", "pinverse": "linear solver", "minimize": "optimiser", "least_squares": "optimiser", "curve_fit": "optimiser",
    "LinearRegression": "regression", "Ridge": "regression", "RidgeCV": "regression", "Lasso": "regression",
    "LogisticRegression": "regression", "SGDClassifier": "fitting", "SGDRegressor": "fitting", "MLPClassifier": "fitting",
    "MLPRegressor": "fitting", "LinearSVC": "fitting", "MultiheadAttention": "simulating the network",
    "TransformerEncoderLayer": "simulating the network",
    "__builtins__": "escape",
}
SOLVER_CALLS = {"solve", "cholesky_solve"}  # a variable may be called solve: these count when called, or as linalg.solve
BUILTINS = {"exec", "eval", "compile", "__import__", "setattr"}  # model.eval() and re.compile are not these
EDITED = re.compile(r"/work/+edited\b")
MAX_PATH_TEXT = 4_096  # a name's tracked path; doubling a string in a few lines must not cost the host memory
WRITE_MODE = re.compile(r"[wax+]")
FIRST_ARG_WRITES = {"save", "savez", "savez_compressed", "savetxt", "remove", "unlink", "rmtree", "mkdir", "makedirs", "rmdir", "truncate"}
SECOND_ARG_WRITES = {"copy", "copy2", "copyfile", "copytree", "symlink", "link", "copyfileobj"}
BOTH_ARG_WRITES = {"move", "rename", "replace"}
RECEIVER_WRITES = {"write_text", "write_bytes", "unlink", "touch", "mkdir", "rmdir", "tofile", "rename", "replace", "symlink_to"}
SHELL_WRITE = re.compile(r"\b(cp|mv|rm|ln|tee|rsync|dd)\b|>")
LOSSY = re.compile(r"loss|objective|fitness")
WEIGHTY = re.compile(r"weight|save_edit")


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    return []


def _identifiers(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id.lower())
        elif isinstance(n, ast.Attribute):
            out.add(n.attr.lower())
    return out


def _strings(node: ast.AST, env: dict[str, str]) -> str:
    """The text a path expression can hold: its constants, joined as a path would be, and those of the names in it."""
    parts = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            parts.append(n.value)
        elif isinstance(n, ast.Name) and n.id in env:
            parts.append(env[n.id])
    return "/".join(p.strip("/") if i else p.rstrip("/") for i, p in enumerate(parts))


def _bind(target: ast.AST, value: str, env: dict[str, str]) -> None:
    for n in ast.walk(target):
        if isinstance(n, ast.Name):
            env[n.id] = value[:MAX_PATH_TEXT]


def _yields_names(it: ast.AST) -> bool:
    """os.listdir gives bare file names: a loop over the edit folder that writes name elsewhere is not writing there."""
    if not isinstance(it, ast.Call):
        return False
    f = it.func
    return (f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else "") in ("listdir", "sorted", "list") \
        and not (isinstance(f, ast.Name) and it.args and not _yields_names(it.args[0]))


def _path_names(tree: ast.AST) -> dict[str, str]:
    """Names assigned something built from a path under the edit folder, in source order, with the loop variable of a walk
    over one. One pass per statement is enough for how paths are written."""
    env: dict[str, str] = {}
    for _ in range(2):
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                text = _strings(n.value, env)
                m = EDITED.search(text)
                if m:
                    for t in n.targets:
                        _bind(t, text[max(m.start() - 256, 0): m.end() + 256], env)
            elif isinstance(n, (ast.For, ast.comprehension)):
                m = EDITED.search(_strings(n.iter, env))
                if m and not _yields_names(n.iter):
                    _bind(n.target, m.group(0), env)
    return env


def _edit_folder_write(node: ast.Call, env: dict[str, str]) -> str | None:
    """The call's text if it writes under /work/edited. Reading from there, as in copying it to a snapshot, is not writing."""
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    args = node.args
    def hit(a) -> bool:
        return bool(EDITED.search(_strings(a, env)))

    mode = next((k.value for k in node.keywords if k.arg == "mode"), None)
    if name == "open":
        receiver = isinstance(f, ast.Attribute) and not (isinstance(f.value, ast.Name) and f.value.id in ("os", "io", "builtins"))
        target, m = (f.value, args[0] if args else mode) if receiver else (args[0] if args else None, args[1] if len(args) > 1 else mode)
        if target is not None and hit(target) and isinstance(m, ast.Constant) and isinstance(m.value, str) and WRITE_MODE.search(m.value):
            return "open for writing"
        return None
    if name in RECEIVER_WRITES and isinstance(f, ast.Attribute) and hit(f.value):
        return name
    if name in FIRST_ARG_WRITES and any(hit(a) for a in args[:2]):  # torch.save takes its path second
        return name
    if name in SECOND_ARG_WRITES and len(args) > 1 and hit(args[1]):
        return name
    if name in BOTH_ARG_WRITES and any(hit(a) for a in args[:2]):
        return name
    if name in ("system", "popen", "run", "Popen", "call", "check_call", "check_output") and args:
        text = _strings(args[0], env)
        if EDITED.search(text) and SHELL_WRITE.search(text):
            return name
    return None


def check_block(code: str) -> tuple[list[str], list[str]]:
    """(refused, flagged). A syntax error refuses nothing: the kernel reports it as any other failure."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [], []
    refused, flagged = set(), set()
    env = _path_names(tree)
    bound = set()  # names this block gives its own meaning: from re import compile, or a variable called eval
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)

    def see_dotted(d: str) -> None:
        refused.update(f"{d} (matches {p})" for p in REFUSED_DOTTED if _matches(d, p))
        flagged.update(f"{why}: {d}" for p, why in FLAGGED_DOTTED.items() if _matches(d, p))

    def see_name(n: str) -> None:
        if n in REFUSED_NAMES:
            refused.add(n)
        if n in FLAGGED_NAMES:
            flagged.add(f"{FLAGGED_NAMES[n]}: {n}")

    for node in ast.walk(tree):
        for m in _modules(node):
            see_dotted(m)
            for part in m.split("."):
                see_name(part)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "linalg":
            for a in node.names:  # an alias, or a name passed to map, is never called by that name
                if a.name in SOLVER_CALLS:
                    flagged.add(f"linear solver: {a.name}")
        if isinstance(node, ast.Attribute):
            see_name(node.attr)
            d = _dotted(node)
            if d:
                see_dotted(d)
                if node.attr in SOLVER_CALLS and d.split(".")[-2:-1] == ["linalg"]:
                    flagged.add(f"linear solver: {node.attr}")
        elif isinstance(node, ast.Name):
            see_name(node.id)
            if node.id in BUILTINS and isinstance(node.ctx, ast.Load) and node.id not in bound:
                flagged.add(f"escape: {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):  # getattr(x, "backward")
            s = node.value.strip()
            if s in REFUSED_NAMES or any(_matches(s, p) for p in REFUSED_DOTTED):
                refused.add(f"{s!r}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2 \
                    and not isinstance(node.args[1], ast.Constant):
                flagged.add("escape: getattr with a computed name")
            called = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if called in SOLVER_CALLS:
                flagged.add(f"linear solver: {called}")
            write = _edit_folder_write(node, env)
            if write:
                flagged.add(f"writing the edit folder directly: {write}")
        elif isinstance(node, (ast.For, ast.While)):
            ids = _identifiers(node)
            if any(LOSSY.search(i) for i in ids) and (any(WEIGHTY.search(i) for i in ids)):
                flagged.add("a loop that updates or stores weights next to a loss")
    return sorted(refused), sorted(flagged)
