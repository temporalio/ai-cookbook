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

4. If discrepancies are found, ask the user which direction to sync:
   - **Update README** → make README match the source file (source is correct)
   - **Update source** → make source file match the README (README is correct, source has unwanted changes)

5. Based on user choice, make the appropriate updates:
   - If updating README: replace snippet content with current source file content
   - If updating source: replace source file content with the README snippet
   - Preserve markdown structure when editing README

## Scope

By default, check **all** READMEs in the repository that contain file annotations (`*File: ...*` patterns).

If the user specifies a path, check only READMEs in that directory.

## Example Output

```
Checking README snippets...

foundations/claim_check_pattern_python/README.md:
  - codec/plugin.py: MISMATCH (source has 23 lines, snippet has 21 lines)
  - codec/claim_check.py: OK

Found 1 mismatch. Which direction should I sync?
  1. Update README to match source (source is correct)
  2. Update source to match README (revert source changes)
```
