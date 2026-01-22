# Check README Snippets

Verify that code snippets in README files match their source files.

## Instructions

1. Find all README.md files in the repository that contain code snippets with file annotations like `*File: path/to/file.py*`

2. For each annotated snippet:
   - Extract the code between the ``` fences
   - Read the actual source file
   - Compare them (exact match for full files, substring for excerpts marked with `(excerpt)`)

3. Report findings:
   - List any mismatches found
   - Show a brief diff or summary of what changed

4. If discrepancies are found, ask the user:
   "Found X mismatches. Would you like me to update the README snippets to match the source files?"

5. If user agrees, update the README files:
   - Replace the code snippet content with the current source file content
   - Preserve the markdown structure (annotations, fences, surrounding text)

## Scope

By default, check only READMEs in directories containing recently modified Python files (use `git status` to find them).

If the user specifies a path or `--all`, check that scope instead.

## Example Output

```
Checking README snippets...

foundations/claim_check_pattern_python/README.md:
  - codec/plugin.py: MISMATCH (source has 23 lines, snippet has 21 lines)
  - codec/claim_check.py: OK

Found 1 mismatch. Would you like me to update the README snippets to match the source files?
```
