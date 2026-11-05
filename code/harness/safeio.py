"""Every host access to a folder the container can write (/work, /ctl) goes through here.

The agent can create links, FIFOs and directories in those folders at any moment, so a checked path can change before it
is used. The rule that follows: never hand such a path to the OS as a whole. Each component is opened relative to its
parent with O_NOFOLLOW, the last one must be a regular file, and nothing is created under a name the agent can predict:
files are built in a host-only folder and moved in with rename, which replaces a link and never follows it.
A root argument is a host-owned folder (a mount source); the container can't replace it."""

import errno
import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK  # O_NONBLOCK: opening a FIFO must not hang the host
CHUNK = 1 << 20
DISK_STEP = 64 << 20  # a copy looks at the free space this often, so it can pass the reserve by at most this

assert shutil.rmtree.avoids_symlink_attacks, "this platform can't remove an agent's tree safely"


class Unsafe(Exception):
    """A path the host won't touch: a link, a FIFO, a name that leaves the root, a file too large."""


class DiskReserve(Unsafe):
    """A copy would have taken the disk below its reserve. Nothing partial is left."""


class Budget:
    """What a copy may write. A file's size at open proves nothing, since a writer in the container can grow it while it
    is read, so every limit is checked against the bytes actually moved, and the disk is checked again as they move."""

    def __init__(self, limit: int | None = None, disk: Path | None = None, reserve: int = 0):
        self.limit, self.disk, self.reserve = limit, disk, reserve
        self.used = 0
        self.next_check = 0

    def check_disk(self) -> None:
        if self.disk is not None and shutil.disk_usage(self.disk).free < self.reserve + DISK_STEP:
            raise DiskReserve(f"less than {(self.reserve + DISK_STEP) // 2 ** 30} GB free on disk")

    def take(self, n: int) -> None:
        self.used += n
        if self.limit is not None and self.used > self.limit:
            raise Unsafe(f"larger than {self.limit} bytes")
        if self.used >= self.next_check:
            self.check_disk()
            self.next_check = self.used + DISK_STEP


def split_rel(rel: str) -> list[str]:
    """Components of a path relative to a root. No absolute paths, no empty components, no dots."""
    if not isinstance(rel, str) or not rel or "\0" in rel or rel.startswith("/"):
        raise Unsafe(f"not a relative path: {rel!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise Unsafe(f"not a plain path: {rel!r}")
    return parts


@contextmanager
def dir_fd(root, parts: list[str] = ()):
    """An fd for root/parts, each component opened from the last with no link followed."""
    fd = os.open(root, DIR_FLAGS)
    try:
        for p in parts:
            try:
                nxt = os.open(p, DIR_FLAGS, dir_fd=fd)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    raise FileNotFoundError(p)
                raise Unsafe(f"{p}: {e.strerror}")
            os.close(fd)
            fd = nxt
        yield fd
    finally:
        os.close(fd)


@contextmanager
def open_read(root, rel: str, limit: int | None = None):
    """A regular file under root, opened for reading. Raises Unsafe for anything else, FileNotFoundError if absent."""
    parts = split_rel(rel)
    with dir_fd(root, parts[:-1]) as d:
        try:
            fd = os.open(parts[-1], FILE_FLAGS, dir_fd=d)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise FileNotFoundError(rel)
            raise Unsafe(f"{rel}: {e.strerror}")  # ELOOP on macOS and Linux for a link
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise Unsafe(f"{rel}: not a regular file")
        if limit is not None and st.st_size > limit:
            raise Unsafe(f"{rel}: larger than {limit} bytes")
        os.set_blocking(fd, True)
        f = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with f:
        yield f


def read_bytes(root, rel: str, limit: int) -> bytes:
    with open_read(root, rel, limit) as f:
        data = f.read(limit + 1)
    if len(data) > limit:  # it grew after the check
        raise Unsafe(f"{rel}: larger than {limit} bytes")
    return data


def read_text(root, rel: str, limit: int) -> str:
    return read_bytes(root, rel, limit).decode("utf-8", "replace")


def tail_text(root, rel: str, chars: int) -> str:
    with open_read(root, rel) as f:
        size = os.fstat(f.fileno()).st_size
        f.seek(max(size - 4 * chars, 0))
        return f.read(4 * chars).decode("utf-8", "replace")[-chars:]


def read_json(root, rel: str, limit: int):
    return json.loads(read_bytes(root, rel, limit))


def sha256_of(root, rel: str, limit: int | None = None) -> tuple[str, int]:
    h = hashlib.sha256()
    budget = Budget(limit)
    with open_read(root, rel, limit) as f:
        while chunk := f.read(CHUNK):
            budget.take(len(chunk))
            h.update(chunk)
    return h.hexdigest(), budget.used


def listdir(root, rel: str | None = None) -> list[str]:
    """Names in root or root/rel; a link there is refused, not listed."""
    with dir_fd(root, split_rel(rel) if rel else []) as d:
        return sorted(os.listdir(d))


def walk(root, prefix: str = ""):
    """(relative path, lstat result) for everything under root, never entering a link."""
    with dir_fd(root, split_rel(prefix) if prefix else []) as d:
        with os.scandir(d) as it:
            entries = sorted(((e.name, e.stat(follow_symlinks=False)) for e in it), key=lambda x: x[0])
    for name, st in entries:
        rel = f"{prefix}/{name}" if prefix else name
        yield rel, st
        if stat.S_ISDIR(st.st_mode):
            try:
                yield from walk(root, rel)
            except (Unsafe, OSError):
                continue  # replaced under us: skip it


def du(root) -> int:
    return sum(st.st_size for _, st in walk(root) if stat.S_ISREG(st.st_mode))


def copy_out(root, rel: str, dst: Path, limit: int | None = None, reserve: tuple[Path, int] | None = None) -> tuple[str, int]:
    """Copy a file from an agent folder to a host path of ours, at most limit bytes, however far it grows meanwhile, and
    never past the disk reserve (folder, bytes). Returns its sha256 and size; on any failure nothing is left behind."""
    budget = Budget(limit, *(reserve or (None, 0)))
    budget.check_disk()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}")
    h = hashlib.sha256()
    try:
        with open_read(root, rel, limit) as f, open(tmp, "xb") as out:
            while chunk := f.read(CHUNK):
                budget.take(len(chunk))
                h.update(chunk)
                out.write(chunk)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return h.hexdigest(), budget.used


def remove(d: int, name: str) -> None:
    """Remove whatever is at name in the folder d: a link is unlinked, a tree removed without following links."""
    try:
        st = os.stat(name, dir_fd=d, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(name, dir_fd=d)
    else:
        os.unlink(name, dir_fd=d)


def publish(src: Path, root, name: str, tries: int = 8) -> None:
    """Move a host-private file or folder to root/name. rename(2) replaces a file or link and follows neither, so
    a planted name can only make it fail; then the name is cleared and it goes again. src must be on root's volume."""
    if "/" in name or name in ("", ".", ".."):
        raise Unsafe(f"not a plain name: {name!r}")
    with dir_fd(root) as d:
        for _ in range(tries):
            try:
                os.rename(src, name, dst_dir_fd=d)
                return
            except OSError as e:
                if e.errno not in (errno.EISDIR, errno.ENOTDIR, errno.ENOTEMPTY, errno.EEXIST):
                    raise
                remove(d, name)
    raise Unsafe(f"{name}: kept being replaced while it was published")
