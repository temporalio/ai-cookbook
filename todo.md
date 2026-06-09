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
- [ ] 1. checks/python/quality.py + shipped ruff/mypy configs
- [ ] 2. Register quality checks
- [ ] 3. Tests (violation → finding; clean → none; missing tool → info)
- [ ] 4. Verify full lint on corpus; exit-code rule holds

### Step 11: Author the Vale ruleset
- [ ] 1. toolkit/.vale.ini + toolkit/styles/AICookbook (HeadingsSentenceCase, MarketingLanguage, FileAnnotation) + vocab
- [ ] 2. toolkit/styles/AICookbook/README.md documenting rules + "earn its place" bar
- [ ] 3. Verify Vale (via --config) fires only on genuine issues; trim noise

### Step 12: review-recipe command + recipe-reviewer agent
- [ ] 1. Author toolkit/agents/recipe-reviewer.md (prose triggers + "When to invoke"; ${CLAUDE_PLUGIN_ROOT} tool paths)
- [ ] 2. Author toolkit/commands/review-recipe.md
- [ ] 3. Verify end-to-end via --plugin-dir on guardrails recipe

### Step 13: Wire toolkit into CI (non-blocking)
- [ ] 1. Add lint-recipes.yml (changed-recipe detection, recipe-lint + vale by repo path, report, warn-not-block, error→block)
- [ ] 2. Confirm frontmatter + python-projects remain hard gates
- [ ] 3. Verify with a draft non-conforming PR

## Phase 2 — Reconcile generation + packaging

### Step 14: Move + point recipe-ify at the SSOT (Angie's file — coordinate)
- [ ] 1. Move to toolkit/commands/recipe-ify.md; add frontmatter; read references; generate canonical README; run recipe-lint after
- [ ] 2. Verify generated recipe is canonical + lint-clean

### Step 15: Move + point recipe-scout at SSOT + wishlist (Angie's file — coordinate)
- [ ] 1. Move to toolkit/commands/recipe-scout.md; add frontmatter; reference references; keep taxonomy + wishlist
- [ ] 2. Verify proposal cards on a sample repo

### Step 16: new-recipe command + template
- [ ] 1. toolkit/templates/recipe-skeleton (canonical, runnable, placeholders)
- [ ] 2. Author toolkit/commands/new-recipe.md (copies skeleton to repo-root category dir)
- [ ] 3. Verify skeleton passes lint + uv sync

### Step 17: Package the toolkit plugin + local-load docs
- [ ] 1. Create toolkit/.claude-plugin/plugin.json (no custom paths needed)
- [ ] 2. Confirm all components under toolkit/ + ${CLAUDE_PLUGIN_ROOT} refs; nothing relies on .claude/ or repo-relative paths
- [ ] 3. Root README: three purposes + `claude --plugin-dir <repo>/toolkit` load instructions
- [ ] 4. Verify plugin loads via --plugin-dir; /help lists commands; content agents/ unaffected

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
