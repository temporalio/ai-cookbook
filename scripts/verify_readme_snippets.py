#!/usr/bin/env python3
"""Verify README code snippets match actual source files.

Standalone usage:
    python scripts/verify_readme_snippets.py [--check-all]

Hook usage (Claude Code):
- PostToolUse on Edit/Write for .py: Warns if edited file's README snippet is stale
- PostToolUse on Edit/Write for .md: Warns if README snippets don't match sources
- PreToolUse on Bash with git commit: Blocks if staged files have stale snippets

Exit codes:
    0: All snippets match (or warning printed for post-edit)
    1: Mismatches found (standalone mode)
    2: Mismatch found (blocks pre-commit)
"""

import difflib
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


def generate_diff(snippet_code: str, source_code: str, file_path: str, max_lines: int = 15) -> str:
    """Generate a unified diff between snippet and source.

    Args:
        snippet_code: Code from README snippet
        source_code: Code from source file
        file_path: Path for diff header
        max_lines: Maximum diff lines to show (0 = unlimited)

    Returns:
        Formatted diff string, truncated if needed
    """
    snippet_lines = snippet_code.strip().splitlines(keepends=True)
    source_lines = source_code.strip().splitlines(keepends=True)

    # Ensure lines end with newline for clean diff
    if snippet_lines and not snippet_lines[-1].endswith('\n'):
        snippet_lines[-1] += '\n'
    if source_lines and not source_lines[-1].endswith('\n'):
        source_lines[-1] += '\n'

    diff = list(difflib.unified_diff(
        snippet_lines, source_lines,
        fromfile=f"README:{file_path}",
        tofile=f"source:{file_path}",
        lineterm=''
    ))

    if not diff:
        return ""

    if max_lines and len(diff) > max_lines:
        truncated = diff[:max_lines]
        remaining = len(diff) - max_lines
        truncated.append(f"\n    ... ({remaining} more lines)\n")
        return ''.join(truncated)

    return ''.join(diff)


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


def compare_snippet_with_diff(snippet: dict, repo_root: Path) -> tuple[bool, str, str]:
    """Compare a snippet against its source file, with diff output.

    Returns:
        (matches, error_message, diff_output)
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
        return False, f"Source file not found: {file_path}", ""

    source_content = full_path.read_text()

    if is_excerpt:
        if code.strip() in source_content:
            return True, "", ""
        return False, f"Excerpt not found in source", ""
    else:
        if code.strip() == source_content.strip():
            return True, "", ""
        diff = generate_diff(code, source_content, file_path)
        return False, "Content mismatch", diff


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


def check_all_readmes(repo_root: Path) -> int:
    """Check all READMEs in the repo for snippet mismatches.

    Returns:
        Exit code (0 = all OK, 1 = mismatches found)
    """
    total_snippets = 0
    total_ok = 0
    total_mismatch = 0
    total_missing = 0

    # Find all READMEs with file annotations
    for readme_path in repo_root.rglob("README.md"):
        try:
            content = readme_path.read_text()
        except Exception:
            continue

        if "*File:" not in content:
            continue

        snippets = extract_snippets(readme_path)
        if not snippets:
            continue

        # Print README header
        rel_readme = readme_path.relative_to(repo_root)
        print(f"\n{rel_readme}:")

        for snippet in snippets:
            total_snippets += 1
            file_path = snippet["file_path"]
            matches, error, diff = compare_snippet_with_diff(snippet, repo_root)

            if matches:
                total_ok += 1
                print(f"  {file_path}: OK")
            elif "not found" in error.lower():
                total_missing += 1
                print(f"  {file_path}: MISSING - {error}")
            else:
                total_mismatch += 1
                print(f"  {file_path}: MISMATCH")
                if diff:
                    # Indent diff output
                    for line in diff.splitlines():
                        print(f"    {line}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total: {total_snippets} snippets checked")
    print(f"  OK: {total_ok}")
    if total_mismatch:
        print(f"  MISMATCH: {total_mismatch}")
    if total_missing:
        print(f"  MISSING: {total_missing}")

    if total_mismatch or total_missing:
        return 1
    return 0


def main():
    # Check for standalone invocation
    if len(sys.argv) > 1 or not os.environ.get("TOOL_NAME"):
        repo_root = Path(subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        ).stdout.strip())

        if not repo_root.exists():
            print("Error: Not in a git repository", file=sys.stderr)
            sys.exit(1)

        print("Checking README snippets...")
        sys.exit(check_all_readmes(repo_root))

    # Hook mode
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


if __name__ == "__main__":
    main()
