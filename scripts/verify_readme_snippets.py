#!/usr/bin/env python3
"""Verify README code snippets match actual source files.

Used as a Claude hook to warn/block when README snippets are out of sync.

Modes:
- PostToolUse on Edit/Write for .py: Warns if edited file's README snippet is stale
- PostToolUse on Edit/Write for .md: Warns if README snippets don't match sources
- PreToolUse on Bash with git commit: Blocks if staged files have stale snippets

Exit codes:
    0: All snippets match (or warning printed for post-edit)
    2: Mismatch found (blocks pre-commit)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def get_tool_input():
    """Get the tool input from environment variable."""
    tool_input = os.environ.get("TOOL_INPUT", "{}")
    try:
        return json.loads(tool_input)
    except json.JSONDecodeError:
        return {}


def get_tool_name() -> str:
    """Get the tool name from environment variable."""
    return os.environ.get("TOOL_NAME", "")


def is_post_edit_hook() -> bool:
    """Check if running as PostToolUse hook on Edit or Write."""
    return get_tool_name() in ("Edit", "Write")


def is_pre_bash_hook() -> bool:
    """Check if running as PreToolUse hook on Bash."""
    return get_tool_name() == "Bash"


def is_git_commit(tool_input: dict) -> bool:
    """Check if the bash command is a git commit."""
    command = tool_input.get("command", "")
    return "git commit" in command


def get_staged_python_files() -> list[str]:
    """Get list of staged Python files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", "*.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def find_readme_for_file(filepath: str) -> Path | None:
    """Find README.md in same directory or parent directories."""
    path = Path(filepath)
    current = path.parent

    while current != current.parent:
        readme = current / "README.md"
        if readme.exists():
            return readme
        current = current.parent

    return None


def extract_snippets(readme_path: Path) -> list[dict]:
    """Extract code snippets with file annotations from README.

    Looks for patterns like:
        *File: path/to/file.py*
        ```python
        code here
        ```

    Or for excerpts:
        *File: path/to/file.py (excerpt)*
    """
    content = readme_path.read_text()
    snippets = []

    pattern = r'\*File:\s*([^\*\(]+?)(?:\s*\(excerpt\))?\s*\*\s*\n+```(?:python)?\n(.*?)```'
    for match in re.finditer(pattern, content, re.DOTALL):
        file_annotation = match.group(0)
        file_path = match.group(1).strip()
        code = match.group(2)
        is_excerpt = "(excerpt)" in file_annotation

        snippets.append({
            "file_path": file_path,
            "code": code,
            "is_excerpt": is_excerpt,
            "readme": readme_path,
        })

    return snippets


def compare_snippet(snippet: dict, repo_root: Path) -> tuple[bool, str]:
    """Compare a snippet against its source file.

    Returns:
        (matches, error_message)
    """
    file_path = snippet["file_path"]
    code = snippet["code"]
    is_excerpt = snippet["is_excerpt"]

    # Resolve relative to README location
    readme_dir = snippet["readme"].parent
    full_path = readme_dir / file_path
    if not full_path.exists():
        full_path = repo_root / file_path

    if not full_path.exists():
        return False, f"Source file not found: {file_path}"

    source_content = full_path.read_text()

    if is_excerpt:
        if code.strip() in source_content:
            return True, ""
        return False, f"Excerpt not found in {file_path}"
    else:
        if code.strip() == source_content.strip():
            return True, ""
        return False, f"Full file mismatch for {file_path}"


def check_single_file(filepath: str, repo_root: Path) -> list[str]:
    """Check a single file against README snippets. Returns list of errors."""
    errors = []
    readme = find_readme_for_file(filepath)
    if not readme:
        return errors

    snippets = extract_snippets(readme)
    file_basename = Path(filepath).name

    for snippet in snippets:
        snippet_basename = Path(snippet["file_path"]).name
        if file_basename == snippet_basename or filepath.endswith(snippet["file_path"]):
            matches, error = compare_snippet(snippet, repo_root)
            if not matches:
                errors.append(f"{readme}: {error}")

    return errors


def check_readme_snippets(readme_path: Path, repo_root: Path) -> list[str]:
    """Check all snippets in a README against their source files."""
    errors = []
    snippets = extract_snippets(readme_path)

    for snippet in snippets:
        matches, error = compare_snippet(snippet, repo_root)
        if not matches:
            errors.append(f"{readme_path}: {error}")

    return errors


def main():
    tool_input = get_tool_input()

    repo_root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    ).stdout.strip())

    # Post-edit hook: warn but don't block
    if is_post_edit_hook():
        file_path = tool_input.get("file_path", "")
        if not file_path:
            sys.exit(0)

        # Python file edited: check if its README snippet is stale
        if file_path.endswith(".py"):
            errors = check_single_file(file_path, repo_root)
            if errors:
                print("Warning: README snippet may be out of sync:")
                for error in errors:
                    print(f"  - {error}")
                print("Run /check-readme-snippets to verify and sync.")
            sys.exit(0)

        # README edited: check if its snippets match source files
        if file_path.endswith(".md") and Path(file_path).name == "README.md":
            errors = check_readme_snippets(Path(file_path), repo_root)
            if errors:
                print("Warning: README snippets may be out of sync with source:")
                for error in errors:
                    print(f"  - {error}")
                print("Run /check-readme-snippets to verify and sync.")
            sys.exit(0)

        sys.exit(0)

    # Pre-commit hook: block if staged files have mismatched snippets
    if is_pre_bash_hook():
        if not is_git_commit(tool_input):
            sys.exit(0)

        staged_files = get_staged_python_files()
        if not staged_files:
            sys.exit(0)

        errors = []
        checked_readmes = set()

        for staged_file in staged_files:
            readme = find_readme_for_file(staged_file)
            if not readme or readme in checked_readmes:
                continue

            checked_readmes.add(readme)
            snippets = extract_snippets(readme)

            for snippet in snippets:
                staged_basename = Path(staged_file).name
                snippet_basename = Path(snippet["file_path"]).name

                if staged_basename == snippet_basename or staged_file.endswith(snippet["file_path"]):
                    matches, error = compare_snippet(snippet, repo_root)
                    if not matches:
                        errors.append(f"{readme}: {error}")

        if errors:
            print("README snippet verification failed:")
            for error in errors:
                print(f"  - {error}")
            print("\nUpdate the README or run /check-readme-snippets to sync.")
            sys.exit(2)

        sys.exit(0)

    # Manual invocation (no TOOL_NAME): this shouldn't happen in normal use
    sys.exit(0)


if __name__ == "__main__":
    main()
