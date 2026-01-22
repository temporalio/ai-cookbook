#!/usr/bin/env python3
"""Verify README code snippets match actual source files.

Used as a Claude hook to block commits when README snippets are out of sync.

Exit codes:
    0: All snippets match (or not a git commit)
    2: Mismatch found (blocks the commit)
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


def is_git_commit(tool_input: dict) -> bool:
    """Check if the bash command is a git commit."""
    command = tool_input.get("command", "")
    return "git commit" in command and "git commit" not in command.replace("git commit", "")


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


def main():
    tool_input = get_tool_input()

    if not is_git_commit(tool_input):
        sys.exit(0)

    staged_files = get_staged_python_files()
    if not staged_files:
        sys.exit(0)

    repo_root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    ).stdout.strip())

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
        print("\nUpdate the README to match the source file, or mark as (excerpt) if partial.")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
