"""Code provenance for the native-C0 checkpoint sweep (PREREG_31_CHECKPOINTS.md, Freeze and 4.3).

The fifth draft of the preregistration freezes by a git commit: the cluster receives the code as
``git archive`` of that commit, every result records the commit, and the analysis script compares
every commit a job ran from with the freeze commit. The fourth draft's addendum, hash chain, step
gates and software lock are gone. What stays here is what makes "the code that ran is the
commit's" checkable:

* ``import_closure``: the static closure of a program's ``import`` statements, used by the
  bundle builder to refuse a bundle that lacks a file the programs import;
* ``check_code_is_from_repo``: the program and every repository package were loaded from the
  ``--repo`` the program was told to run from, not from a second copy elsewhere on ``sys.path``;
* ``code_record``: the sha256 of every repository file loaded so far. Each manifest and result
  records it; a resumed invocation refuses if a Python file it loaded differs from the one the
  first invocation loaded; and the analysis script checks every recorded hash against
  ``git show <commit>:<path>`` for the commit the result names.
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from spatialvlm.eval.native_checkpoint_allowlist import refuse


def file_sha256(path: str | Path) -> str:
    """Return the streaming sha256 of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── static import closure ─────────────────────────────────────────────────────────────────────

# Where Python finds repository code inside the image, in sys.path order: the working
# directory (the repository root, for ``python -m``), then PYTHONPATH (``src`` and the vendored
# GATr checkout). Roots that do not exist are skipped.
IMPORT_ROOTS: tuple[str, ...] = (".", "src", "REPOS/geometric-algebra-transformer")
# Top-level packages whose code lives in the repository. Any loaded module of these packages must
# come from inside --repo; the first two must be loaded for the program to be running at all.
REPOSITORY_PACKAGES: tuple[str, ...] = ("scripts", "spatialvlm", "gatr")
REQUIRED_PACKAGES: tuple[str, ...] = ("scripts", "spatialvlm")


def _module_file(name: str, roots: Sequence[Path]) -> Path | None:
    parts = name.split(".")
    for root in roots:
        base = root.joinpath(*parts)
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
        candidate = base.with_suffix(".py")
        if candidate.is_file():
            return candidate
    return None


def _package_inits(path: Path) -> list[Path]:
    """The ``__init__.py`` files of the packages that enclose ``path``, innermost first.

    A directory is a package only if it holds ``__init__.py``; the walk stops at the first one
    that does not (``src/`` and ``tests/`` do not), exactly as Python names the module.
    """
    inits: list[Path] = []
    directory = path.parent
    while (directory / "__init__.py").is_file():
        inits.append(directory / "__init__.py")
        directory = directory.parent
    return inits


def _module_name(path: Path) -> tuple[str, bool]:
    """Name a file as Python does: ``src/spatialvlm/eval/x.py`` is ``spatialvlm.eval.x``."""
    is_package = path.name == "__init__.py"
    parts = [init.parent.name for init in reversed(_package_inits(path))]
    if not is_package:
        parts.append(path.stem)
    return ".".join(parts), is_package


def _imported_names(tree: ast.AST, module: str, is_package: bool) -> Iterable[str]:
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".") if package else []
                if node.level > 1:
                    anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *(node.module.split(".") if node.module else [])])
            else:
                base = node.module or ""
            if base:
                yield base
            for alias in node.names:
                if alias.name != "*":
                    yield f"{base}.{alias.name}" if base else alias.name


def import_closure(
    entry_files: Sequence[str | Path],
    *,
    repo: Path,
    roots: Sequence[str | Path] = IMPORT_ROOTS,
) -> list[str]:
    """Return every repository file reachable from ``entry_files`` by ``import`` statements.

    Every ``import`` statement in a file is followed, including those inside functions, and every
    package ``__init__.py`` that importing a dotted name runs is included. A dotted name resolves
    against ``roots`` in order, as Python's ``sys.path`` does; names that do not resolve to a file
    under a root inside ``repo`` are outside the repository. Non-Python entries are listed as they
    are. Imports built at run time are invisible here; ``code_record`` sees them.
    """
    repo = repo.resolve()
    search_roots = [(repo / root).resolve() for root in roots if (repo / root).is_dir()]
    pending = [(repo / Path(item)).resolve() for item in entry_files]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Import-closure entry is missing: {path}")
        seen.add(path)
        if path.suffix != ".py":
            continue
        module, is_package = _module_name(path)
        # Running a module inside a package runs every enclosing package __init__.py first.
        for init in _package_inits(path):
            if init.resolve() not in seen:
                pending.append(init.resolve())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree, module, is_package):
            pieces = name.split(".")
            for depth in range(1, len(pieces) + 1):
                target = _module_file(".".join(pieces[:depth]), search_roots)
                if target is not None and target.is_relative_to(repo) and target not in seen:
                    pending.append(target)
    return sorted(str(path.relative_to(repo)) for path in seen)


def repository_modules(
    repo: Path, modules: Mapping[str, ModuleType | None] | None = None
) -> dict[str, str]:
    """Return {module name: repository-relative file} for every loaded module inside ``repo``.

    Only file-backed modules count. Python gives every module it loads from a file an absolute
    ``__file__``; a relative one belongs to a synthetic namespace object (``torch.classes`` is
    ``_classes.py`` and ``torch.ops`` is ``_ops.py``) or to ``<stdin>``, and resolving it against
    the working directory would name a repository file that does not exist. A relative value is
    therefore counted only if it names an existing file.
    """
    repo = repo.resolve()
    loaded = dict(sys.modules if modules is None else modules)
    result: dict[str, str] = {}
    for name, module in loaded.items():
        filename = getattr(module, "__file__", None) if module is not None else None
        if not isinstance(filename, str) or not filename:
            continue
        if not os.path.isabs(filename) and not Path(filename).is_file():
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(repo):
            result[name] = str(path.relative_to(repo))
    return result


def check_code_is_from_repo(
    repo: Path,
    *,
    entry_file: str | Path,
    expected_entry: str,
    packages: Sequence[str] = REPOSITORY_PACKAGES,
    required: Sequence[str] = REQUIRED_PACKAGES,
    modules: Mapping[str, ModuleType | None] | None = None,
) -> dict[str, str]:
    """Refuse unless the code that is running is the code under ``repo``.

    A code record of the files under ``--repo`` means nothing if Python loaded the program or a
    repository package from a second copy elsewhere on ``sys.path``. So: the entry file must be
    ``repo / expected_entry``, every loaded module of a repository package (``scripts``,
    ``spatialvlm``, ``gatr``) must come from inside ``repo``, and the required packages must be
    loaded at all. Returns {package: file} for the required packages.
    """
    repo = repo.resolve()
    entry = Path(entry_file).resolve()
    if entry != (repo / expected_entry).resolve():
        raise refuse(f"the program runs from {entry}, not from --repo {repo}.")
    loaded = dict(sys.modules if modules is None else modules)
    outside: dict[str, str] = {}
    found: dict[str, str] = {}
    for name, module in loaded.items():
        if name.split(".")[0] not in packages or module is None:
            continue
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str) or not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(repo):
            outside[name] = str(path)
        elif name in required:
            found[name] = str(path.relative_to(repo))
    if outside:
        raise refuse(f"repository packages were loaded from outside --repo {repo}: {outside}")
    absent = [name for name in required if name not in found]
    if absent:
        raise refuse(f"required repository packages are not loaded from --repo {repo}: {absent}")
    return found


def code_record(
    repo: Path,
    *,
    entry_file: str | Path,
    expected_entry: str,
    modules: Mapping[str, ModuleType | None] | None = None,
) -> dict[str, str]:
    """{repository path: sha256} of every repository file loaded so far, sorted by path.

    Refuses first unless the program and its packages run from ``repo``
    (``check_code_is_from_repo``).
    """
    check_code_is_from_repo(
        repo, entry_file=entry_file, expected_entry=expected_entry, modules=modules
    )
    repo = repo.resolve()
    files = sorted(set(repository_modules(repo, modules).values()))
    return {relative: file_sha256(repo / relative) for relative in files}


def check_code_unchanged(first: Mapping[str, str], now: Mapping[str, str]) -> None:
    """Refuse if a repository file loaded by both invocations differs between them.

    Files loaded only now (a later import) are allowed; their hashes go into the new record.
    """
    changed = sorted(path for path in set(first) & set(now) if first[path] != now[path])
    if changed:
        raise refuse(
            f"repository files differ from the ones the first invocation loaded: {changed}; "
            "after the freeze commit only result files and the kit's shell scripts may change."
        )


def code_record_sha256(record: Mapping[str, Any]) -> str:
    """One hash for a code record, for invocation lines and episode reports."""
    data = "".join(f"{digest}  {path}\n" for path, digest in sorted(record.items()))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
