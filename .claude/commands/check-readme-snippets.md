# Check README Snippets

Verify that code snippets in README files match their source files.

## Instructions

**CRITICAL: You MUST run the verification script first. Do NOT manually explore or compare files.**

1. Run the verification script:
   ```bash
   uv run python scripts/verify_readme_snippets.py
   ```

2. Report the results to the user. The script shows:
   - OK/MISMATCH/MISSING status for each snippet
   - Unified diffs for mismatches
   - Summary counts

3. If all snippets are OK, report success and stop.

4. If mismatches are found, ask the user which direction to sync:
   - **Update README** → make README match the source file
   - **Update source** → make source file match the README

5. Apply the user's chosen fix by editing the appropriate file(s).
