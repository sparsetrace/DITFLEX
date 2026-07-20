#!/usr/bin/env python
"""context/context.py -- snapshot the repo into one shareable markdown.
======

Walks the repo, writes context/context.md containing every source file in
a fenced code block with its path as a header, preceded by a file tree.
Intended to run on a plain GitHub Actions runner (no GPU, no Modal, no
dependencies beyond the stdlib) via .github/workflows/context.yml, or
locally:

    python context/context.py

Deterministic output: same tree -> same file, so the commit step can
detect "no changes" and skip.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUT_FILE = Path(__file__).parent / "context.md"

# Extensions worth including, mapped to a fence language hint.
LANGS = {
    ".py": "python",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".cfg": "ini",
    ".ini": "ini",
    ".sh": "bash",
    ".txt": "",
    ".json": "json",
}

# Directories never worth walking into.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "*.egg-info",
}

# Individual files to skip.
#  - the output itself MUST be here, or it embeds its previous self and
#    doubles on every run
#  - this generator is tooling, not the program: it does not belong in
#    a snapshot of the ditflex source
SKIP_FILES = {OUT_FILE.resolve(), Path(__file__).resolve()}

MAX_BYTES = 512 * 1024  # listed but not inlined beyond this


def should_skip_dir(path: Path) -> bool:
    name = path.name
    return name in SKIP_DIRS or name.endswith(".egg-info")


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(should_skip_dir(parent) for parent in path.parents):
            continue
        if path.resolve() in SKIP_FILES:
            continue
        if path.suffix.lower() not in LANGS:
            continue
        files.append(path)
    return files


def fence_for(text: str) -> str:
    """A backtick fence strictly longer than any backtick run in the
    content, minimum 3 -- so embedded ``` (in markdown files, docstrings)
    cannot terminate the block early."""
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def render_tree(files: list[Path], root: Path) -> str:
    lines = [f"{root.name}/"]
    seen_dirs: set[tuple[str, ...]] = set()
    for f in files:  # sorted, so directories appear before their contents
        rel = f.relative_to(root)
        for i in range(1, len(rel.parts)):
            d = rel.parts[:i]
            if d not in seen_dirs:
                seen_dirs.add(d)
                lines.append("    " * (i - 1) + "├── " + d[-1] + "/")
        lines.append("    " * (len(rel.parts) - 1) + "├── " + rel.name)
    return "\n".join(lines)


def main() -> int:
    files = collect_files(REPO_ROOT)
    if not files:
        print("no source files found -- wrong root?", file=sys.stderr)
        return 1

    parts: list[str] = []
    parts.append("# ditflex -- repo snapshot\n")
    parts.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"by context/context.py. {len(files)} files.\n"
    )
    parts.append("## Tree\n")
    parts.append("```\n" + render_tree(files, REPO_ROOT) + "\n```\n")
    parts.append("## Files\n")

    for path in files:
        rel = path.relative_to(REPO_ROOT)
        size = path.stat().st_size
        parts.append(f"### `{rel}`\n")

        if size > MAX_BYTES:
            parts.append(f"*({size:,} bytes -- too large to inline, listed only)*\n")
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            parts.append("*(not valid UTF-8 -- skipped)*\n")
            continue

        fence = fence_for(text)
        lang = LANGS.get(path.suffix.lower(), "")
        body = text if text.endswith("\n") else text + "\n"
        parts.append(f"{fence}{lang}\n{body}{fence}\n")

    OUT_FILE.write_text("\n".join(parts), encoding="utf-8")
    total = OUT_FILE.stat().st_size
    print(f"wrote {OUT_FILE.relative_to(REPO_ROOT)}: {len(files)} files, {total:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
