# Check README Snippets

Verify that code snippets in README files match their source files.

## Instructions

1. **Run the verification script** to get a summary with diffs:
   ```bash
   uv run python scripts/verify_readme_snippets.py
   ```

2. **Review the output** - The script will show:
   - Each README with file annotations
   - OK/MISMATCH/MISSING status for each snippet
   - Unified diffs for mismatches (truncated to 15 lines)
   - Summary counts at the end

3. **If mismatches are found**, ask the user which direction to sync:
   - **Update README** → make README match the source file (source is correct)
   - **Update source** → make source file match the README (README is correct, source has unwanted changes)

4. **Based on user choice**, make the appropriate updates:
   - If updating README: replace snippet content with current source file content
   - If updating source: replace source file content with the README snippet
   - Preserve markdown structure when editing README

## Example Output

```
Checking README snippets...

foundations/claim_check_pattern_python/README.md:
  codec/claim_check.py: OK
  codec/plugin.py: MISMATCH
    --- README:codec/plugin.py
    +++ source:codec/plugin.py
    @@ -1,5 +1,6 @@
     import os
    +from temporalio.plugin import SimplePlugin
     from temporalio.converter import DataConverter

============================================================
Total: 12 snippets checked
  OK: 11
  MISMATCH: 1
```
