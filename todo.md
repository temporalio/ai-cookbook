# TODO: AI Cookbook Authoring & Consistency Toolkit

Mirrors `plan.md`. Check off sub-steps as they complete.

**Plugin mechanics (apply to all plugin steps):** plugin rooted in `toolkit/`, loaded via
`claude --plugin-dir <repo>/toolkit` (local, not published); components at the toolkit root
auto-discover; `.claude/` is project-local only; intra-plugin refs use `${CLAUDE_PLUGIN_ROOT}`,
CI uses repo paths.

## Phase 0 — Foundation: single source of truth + validator fix

### Step 1: Scaffold the recipe-writing skill
- [x] 1. Create toolkit/skills/recipe-writing/ + four reference stubs
- [x] 2. Write SKILL.md (3rd-person trigger desc + read-references-first + tool invocations via ${CLAUDE_PLUGIN_ROOT})
- [x] 3. Stub reference files with H1 + TODO placeholder
- [x] 4. Verify paths exist + SKILL.md frontmatter valid (live --plugin-dir load deferred to Step 17, when plugin.json exists)

### Step 2: Author references/structure.md (canonical README walkthrough)
- [x] 1. Document the mandatory shape + light code-sandwich + H1 rule + allowed variation
- [x] 2. Include a correct example excerpt from hello_world_openai_responses
- [x] 3. Verify against 3 walkthrough recipes; note deviations for Phase 3

### Step 3: Author references/layout.md (layout + slug/URL contract)
- [x] 1. Document stable core, optional dirs, mandatory tests, naming, slug/URL contract, standardize-away
- [x] 2. Decide + document the __init__.py convention (optional; if present, must be empty — lint flags non-empty, not missing)
- [x] 3. Verify against all 13 recipe layouts; list deviations (package-name dups, stray entry files, __init__.py gaps, MCP variant)

### Step 4: Author references/frontmatter.md (schema + governed values)
- [x] 1. Document README-only, schema, HTML-comment rationale, spacing, ordering, accept-list, priority bands, exclusions, YAML validity
- [x] 2. Inventory all tags; produce canonical accept-list + synonym map as tags.json (source of truth)
- [x] 3. Verify; list per-recipe violations (Step 18 worklist)

### Step 5: Author references/code-conventions.md (Temporal rules + quality bar)
- [x] 1. Document each Python rule with rationale + tiny example
- [x] 2. Mark each rule mechanically-checkable vs judgment-only
- [x] 3. Verify rules against two real recipes (guardrails + hello_world — all 13 rules confirmed real/current)

### Step 6: Rewrite the front-matter validator (real YAML, tiered severity)
- [x] 1. Add .github/scripts/package.json (js-yaml) + npm ci/test/install in CI workflow
- [x] 2. Rewrite validator (ESM, js-yaml, hard-error vs warning tiers, accept-list from tags.json, ordering, exclusions, docs-sync parity)
- [x] 3. Write validator tests (node:test, 13 tests) + npm test script + run in CI; all pass
- [x] 4. Reconcile CONTRIBUTING.md + CLAUDE.md claims (description hard-required; tags/priority warnings)
- [x] 5. Verify against whole repo: exit 0, 14 warnings matching the Step-18 worklist

## Phase 1 — Validation tooling

### Step 7: Scaffold the recipe-lint CLI
- [x] 1. Create toolkit/tools/recipe-lint uv package + entry point + dirs (src layout, hatchling, ruff/mypy/pytest config)
- [x] 2. findings.py (Finding, format_report, exit-code rule)
- [x] 3. dispatch.py (detect_language, run_checks; empty CHECKS registry for now)
- [x] 4. cli.py (argparse, text/json)
- [x] 5. Tests for dispatch + findings (9 tests pass)
- [x] 6. Verify: recipe-lint on guardrails → empty/exit 0; ruff + mypy --strict clean

### Step 8: Structural / layout / naming / link checks
- [x] 1. checks/python/structure.py (required files/dirs, mandatory tests, empty-init, naming, task-queue, stray-entry, frontmatter, links/assets; MCP variant handled)
- [x] 2. Register module for "python" (lazy load in dispatch; empty __init__.py preserved)
- [x] 3. Tests with good/bad fixture recipes (test_structure_checks.py)
- [x] 4. Run on all 13 recipes: 0 errors, no crashes; warnings = Phase-3 worklist

### Step 9: Code-convention AST checks
- [x] 1. checks/python/conventions.py (max_retries SCOPED to clients, data converter, timeout, stale models; ApplicationError deferred to reviewer agent per "earn every rule")
- [x] 2. Register checks (added conventions to lazy dispatch import)
- [x] 3. Tests with code-snippet fixtures incl. RetryPolicy-never-flagged (Mason's max_retries constraint)
- [x] 4. Run on corpus: findings accurate/low-FP on sample of 3; ruff+mypy clean; 28 tests pass

### Step 10: Wire ruff + mypy into recipe-lint
- [x] 1. checks/python/quality.py + shipped configs/ruff.toml (mypy deliberately NOT run cross-corpus — untyped SDK noise; documented)
- [x] 2. Register quality checks (added quality to lazy dispatch import)
- [x] 3. Tests (ruff violation → finding; clean → none; missing tool → graceful warning)
- [x] 4. Verify full lint on corpus: coherent report, exit-code rule holds (guardrails → 9 ruff warnings, exit 0)

### Step 11: Author the Vale ruleset
- [x] 1. toolkit/.vale.ini + toolkit/styles/AICookbook + vocab; MarketingLanguage kept; HeadingsSentenceCase dropped (49 FPs — corpus is Title Case); FileAnnotation → reviewer agent (not Vale-expressible)
- [x] 2. toolkit/styles/AICookbook/README.md documenting the rule + "earn its place" bar + the two rejected candidates
- [x] 3. Verify Vale (via --config): 3 genuine MarketingLanguage hits across corpus, exit 0; noise trimmed

### Step 12: review-recipe command + recipe-reviewer agent
- [x] 1. Author toolkit/agents/recipe-reviewer.md (prose triggers + "When to invoke"; reads 4 refs; runs recipe-lint/vale/tests; judgment checks; ${CLAUDE_PLUGIN_ROOT} paths)
- [x] 2. Author toolkit/commands/review-recipe.md (frontmatter + launches reviewer agent + combined report)
- [x] 3. Frontmatter validated; live --plugin-dir end-to-end run deferred to Step 17 (no plugin.json yet)

### Step 13: Wire toolkit into CI (non-blocking)
- [x] 1. Add lint-recipes.yml (changed-recipe detection + matrix; recipe-lint gates on error, Vale advisory; findings → job summary)
- [x] 2. Confirm frontmatter + python-projects workflows remain hard gates (untouched; all 3 parse OK)
- [x] 3. Verify gating locally (warnings→exit 0; missing tests→exit 1); YAML valid. Live non-conforming-PR check is manual (no PR opened this run)

## Phase 2 — Reconcile generation + packaging

### Step 14: Move + point recipe-ify at the SSOT (Angie's file — coordinate)
- [x] 1. Moved to toolkit/commands/recipe-ify.md; added frontmatter; references the SSOT (no inline conventions); canonical walkthrough README; runs recipe-lint after; old .claude/commands/recipe-ify.md removed
- [x] 2. Live /recipe-ify run deferred to Step 17 (--plugin-dir); generated recipes are gated by recipe-lint per the command

### Step 15: Move + point recipe-scout at SSOT + wishlist (Angie's file — coordinate)
- [x] 1. Moved to toolkit/commands/recipe-scout.md; added frontmatter; references SSOT for "good recipe"; kept AI-building-block taxonomy + wishlist; proposal cards reference canonical structure; old file removed
- [x] 2. Live /recipe-scout run deferred to Step 17 (--plugin-dir); command structure validated

### Step 16: new-recipe command + template
- [x] 1. toolkit/templates/recipe-skeleton (runnable, ruff-clean, canonical walkthrough README, ALL_CAPS placeholders)
- [x] 2. Author toolkit/commands/new-recipe.md (asks name/category/provider; copies skeleton to repo-root category dir; fills placeholders)
- [x] 3. Verify: instantiated skeleton → recipe-lint clean (0 findings, exit 0)

### Step 17: Package the toolkit plugin + local-load docs
- [x] 1. Create toolkit/.claude-plugin/plugin.json (name ai-cookbook-toolkit; no custom paths)
- [x] 2. Confirmed: 4 commands + recipe-reviewer agent + recipe-writing skill all under toolkit/; intra-plugin refs use ${CLAUDE_PLUGIN_ROOT}; repo-root agents/ untouched
- [x] 3. Root README: three purposes + `claude --plugin-dir <repo>/toolkit` load instructions
- [x] 4. Structure verified (manifest valid, components discoverable, no agents/ collision); live --plugin-dir /help check is manual

## Phase 3 — Backfill consistency

### Step 18: Backfill front matter (no renames)
- [ ] 1. Edit each README front matter to schema (spacing, vocab, ordering, priority)
- [ ] 2. Run validator per file
- [ ] 3. Verify zero warnings across corpus

### Step 19: Converge README structure to canonical (outliers)
- [ ] 1. Restructure non-canonical READMEs (exclude guardrails + human_in_the_loop)
- [ ] 2. Run vale (--config) + recipe-lint per recipe
- [ ] 3. Verify structure rules pass + render check

### Step 20: Backfill code quality
- [ ] 1. Apply quality bar per recipe (ruff/mypy/conventions), minimal changes
- [ ] 2. Ensure tests still pass
- [ ] 3. Verify no error-severity findings across corpus

### Step 21: Final integration
- [ ] 1. Update CONTRIBUTING.md + CLAUDE.md (toolkit + --plugin-dir load step)
- [ ] 2. Full toolchain dry-run on fresh clone (validator/lint/vale/tests)
- [ ] 3. Verify CI green; plugin loads via --plugin-dir; new-recipe → review-recipe cycle works

## Open items (tracked separately, not part of the 21 steps)

- [ ] Deliver a PR reconciling `human_in_the_loop_python` to canonical, with rationale (outcome may sanction a richer variant)
- [ ] Temporal-wide tag vocabulary across all content properties (Mason owns)
