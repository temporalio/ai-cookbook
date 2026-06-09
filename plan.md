# Plan: AI Cookbook Authoring & Consistency Toolkit

Implementation blueprint for `spec.md`. Build-and-verify steps (no TDD ceremony). The two
real code artifacts — the front-matter validator and the `recipe-lint` CLI — ship with
tests; everything else is authoring (markdown references, Vale rules, plugin config) and
is verified by running the tooling against the existing recipe corpus.

## Current Status

- [ ] **Phase 0 — Foundation** (steps 1–6): single source of truth + validator fix
- [ ] **Phase 1 — Validation tooling** (steps 7–13): recipe-lint, Vale, review command, CI
- [ ] **Phase 2 — Reconcile generation + packaging** (steps 14–17): SSOT wiring, plugin
- [ ] **Phase 3 — Backfill consistency** (steps 18–21): converge the corpus

**Branch:** `ai-cookbook-style`. Steps 1–13 add new files only and do not touch Angie's
`recipe-scout.md` / `recipe-ify.md` / guardrails recipe. Steps 14–15 move and modify those
command files and must be coordinated with Angie per the branch-flow constraint.

## Plugin mechanics (read before any plugin step)

These constraints come from the Claude Code plugin model and govern Steps 1, 11, 12, 14–17.

- **Distribution is local, via `--plugin-dir`.** This is not a published plugin. It is
  loaded with `claude --plugin-dir <repo>/toolkit`. No marketplace entry.
- **The plugin is rooted in `toolkit/`, not the repo root.** Claude Code auto-discovers
  `commands/`, `agents/`, and `skills/` at the **plugin root**. The cookbook already uses
  `agents/` as a recipe **category** at the repo root. Rooting the plugin in `toolkit/`
  keeps the recipe `agents/` directory outside the plugin's component tree, so there is no
  discovery collision — and no need to probe whether discovery recurses.
- **Custom component paths in `plugin.json` supplement defaults; they do not replace
  them.** This is why we relocate rather than try to "redirect" `agents/`.
- **`.claude/` is project-local, not a plugin component path.** Files under `.claude/`
  load for anyone working in the repo, but are NOT discovered as plugin components. The
  durable home for toolkit components is `toolkit/…`. Local `.claude/` skills remain
  acceptable for one-off use.
- **Plugin components reference their own files via `${CLAUDE_PLUGIN_ROOT}/…`** (resolves
  to the `--plugin-dir` target). CI and contributors invoke the same tools by repo path
  (`toolkit/…`). Never hardcode absolute paths.
- **Component layout under `toolkit/`:**
  - `toolkit/.claude-plugin/plugin.json` (manifest)
  - `toolkit/skills/recipe-writing/SKILL.md` + `references/` (the single source of truth)
  - `toolkit/commands/*.md` (recipe-ify, recipe-scout, review-recipe, new-recipe)
  - `toolkit/agents/recipe-reviewer.md`
  - `toolkit/templates/recipe-skeleton/`
  - `toolkit/styles/AICookbook/` + `toolkit/.vale.ini`
  - `toolkit/tools/recipe-lint/` (standalone uv CLI; bundled, not a plugin component)
- **Repo-root, outside the plugin:** the recipe categories (`agents/`, `foundations/`,
  `deep_research/`, `mcp/`) and CI (`.github/`). The validator stays at
  `.github/scripts/validate-frontmatter.js`.

---

## Phase 0 — Foundation: single source of truth + validator fix

### Step 1: Scaffold the recipe-writing skill

**NOTE:** Home for the single source of truth, under the `toolkit/` plugin root. Nothing
references it yet; later steps and the validator/linter/commands all point here.

```text
1. Create the skill directory structure under the toolkit plugin root:
   - toolkit/skills/recipe-writing/SKILL.md
   - toolkit/skills/recipe-writing/references/structure.md       (stub)
   - toolkit/skills/recipe-writing/references/layout.md           (stub)
   - toolkit/skills/recipe-writing/references/frontmatter.md       (stub)
   - toolkit/skills/recipe-writing/references/code-conventions.md  (stub)

2. Write toolkit/skills/recipe-writing/SKILL.md:
   - YAML frontmatter: `name: recipe-writing`; third-person `description` with specific
     trigger phrases: "write a recipe", "review a recipe", "check a recipe against the
     cookbook style", "cookbook-ify", "validate recipe front matter", "recipe conventions".
   - Lean body (imperative voice, written FOR Claude): instruct it to READ the four
     references before writing or reviewing any recipe and never rely on memory; list each
     reference with a one-line summary, addressed as `${CLAUDE_PLUGIN_ROOT}/skills/
     recipe-writing/references/<file>`.
   - Document the two tools and how to run them by plugin root:
     `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint <recipe-dir>`
     and `vale --config ${CLAUDE_PLUGIN_ROOT}/.vale.ini <file-or-dir>` — noting both are
     advisory and the skill is the enforcer that combines their output with judgment.

3. Each stub reference file: a single H1 + a one-line "TODO: authored in Step N"
   placeholder, so cross-links resolve.

4. Verify: load the toolkit locally with `claude --plugin-dir <repo>/toolkit` and confirm
   the skill appears in the available-skills list; confirm all four reference paths exist.
```

### Step 2: Author `references/structure.md` (canonical README walkthrough)

**NOTE:** The canonical "light code-walkthrough" confirmed against the live docs render.
Enforced by `recipe-ify`, `new-recipe`, Vale, and the reviewer agent.

```text
1. Write toolkit/skills/recipe-writing/references/structure.md documenting the canonical
   README:
   - The mandatory shape: H1 title → 1–2 sentence intro → optional "key design decisions"
     bullets OR "## Application Components" → a sequence of "## Create the {Component}"
     sections → "## Running".
   - The light code-sandwich rule: each "Create the X" section introduces the file (what +
     why), shows the code under a `*File: path*` line, then explains key lines/gotchas.
   - The mandatory-H1 rule and WHY (docs sync promotes H1 to the page title; missing H1 is
     a hard docs-build error).
   - Allowed variation: trivial single-file recipes may collapse sections but must keep the
     H1, an intro, and "## Running".
   - Explicitly mark the brief "overview + run + expected output" style as non-canonical.

2. Include a short, correct example excerpt (drawn from
   foundations/hello_world_openai_responses_python/README.md) showing one "Create the X"
   section done right.

3. Verify: re-read against 3 existing walkthrough recipes (hello_world_openai_responses,
   hello_world_litellm, tool_call_openai) and confirm the documented shape matches what
   they actually do. Note any recipe that deviates as a Phase-3 backfill target.
```

### Step 3: Author `references/layout.md` (directory conventions + slug/URL contract)

```text
1. Write toolkit/skills/recipe-writing/references/layout.md documenting:
   - The stable core every recipe has: pyproject.toml, README.md, worker.py,
     start_workflow.py, activities/, workflows/, tests/.
   - Optional dirs and when to use them: models/, tools/, helpers/, agents/, util/,
     codec/, shared/, mcp_servers/, _assets/ (images).
   - TESTS ARE MANDATORY: every recipe ships tests/, even if it must mock the LLM/API so
     tests run without credentials. No recipe is accepted without tests.
   - Naming: directory `{recipe-name}_python`; package `cookbook-{recipe-name}-python`;
     task queue `{recipe-name}-task-queue`; PascalCase workflow class; snake_case activities.
   - The slug/URL contract: the directory name IS the permanent public docs URL. Renames
     are breaking and require a coordinated SLUG_ALIASES entry upstream — never rename
     casually.
   - Standardize-away list: no stray top-level entry files duplicating start_workflow.py
     (hello_world.py, claude_test.py); pick one __init__.py convention (document which).

2. Decide and document the __init__.py convention (recommend: rely on pythonpath=["."]
   and omit __init__.py in recipe subpackages, matching the majority; state it explicitly).

3. Verify: cross-check the documented core/optional layout against the file-layout
   inventory of all 13 recipes; list any deviations as Phase-3 targets.
```

### Step 4: Author `references/frontmatter.md` (schema + governed values)

```text
1. Write toolkit/skills/recipe-writing/references/frontmatter.md documenting:
   - Front matter lives ONLY on README.md, never in code.
   - The schema (description, tags, priority) with the exact HTML-comment form.
   - WHY HTML comment not `---` YAML: invisible in GitHub README render; the docs sync
     parses the comment; we don't change the sync. State this is a fixed constraint.
   - Spacing rule: `tags: [ ... ]` with one space after the colon.
   - Tag ordering: category, then language, then provider.
   - The cookbook tag accept-list (resolve `claude`→`anthropic` synonyms; decide canonical
     forms for the one-offs toolcalling, claim-check, provider-neutral, s3, workflows).
     Enumerate the accepted vocabulary explicitly.
   - Priority bands derived from EXISTING values (no churn): document the current relative
     ordering (foundations highest, then agents, then advanced) as ranges, mapping the
     values already in use.
   - Exclusions: never include `last_updated` or `title` (both derived by the docs sync).
   - Validity: must parse under real YAML (js-yaml) — the docs build uses it.

2. Build the accept-list by inventorying every tag currently used across all recipe
   READMEs; produce the canonical set + a synonym→canonical map. Define this as the single
   source of truth the validator (Step 6) imports — write it as a small JSON the validator
   can read (e.g. toolkit/skills/recipe-writing/references/tags.json) and describe it in
   frontmatter.md. Decide and document which file is authoritative.

3. Verify: list, per existing recipe, which front-matter rules it currently violates
   (spacing, vocab, ordering). This becomes the Step 18 worklist.
```

### Step 5: Author `references/code-conventions.md` (Temporal rules + quality bar)

```text
1. Write toolkit/skills/recipe-writing/references/code-conventions.md documenting the
   Python code-layer rules, each with a one-line rationale and a tiny correct/incorrect
   example:
   - LLM clients constructed with max_retries=0 (Temporal owns retries).
   - pydantic_data_converter in Client.connect() AND the test WorkflowEnvironment.
   - Every workflow.execute_activity call specifies start_to_close_timeout.
   - API-boundary errors caught → ApplicationError(..., non_retryable=True).
   - Naming (cross-reference layout.md): task queue, package, classes, activities.
   - requires-python ">=3.10,<3.14"; temporalio ">=1.15.0,<2".
   - Tests: @pytest.mark.asyncio + @pytest.mark.timeout(30), time-skipping env, mocked
     activities (no real API key); tests mandatory (cross-reference layout.md).
   - Workflows are pure orchestration (no I/O / LLM calls).
   - Current model names (e.g. claude-sonnet-4-6) — flag stale names.
   - Quality bar from the `python` skill: ruff clean (lint+format), strict type checking
     (mypy --strict or equivalent), modern Pythonic style.

2. For each rule, note whether it is mechanically checkable (so Step 8–10 know what to
   implement) vs. judgment-only (so the reviewer agent knows to check it).

3. Verify: spot-check the rules against agents/guardrails_hard_rules_python and
   foundations/hello_world_openai_responses_python; confirm each rule is real and current.
```

### Step 6: Rewrite the front-matter validator (real YAML, tiered severity)

**NOTE:** `.github/scripts/validate-frontmatter.js` stays at the repo root (it is CI, not a
plugin component). It currently uses a homegrown parser and hard-fails on any error. Align
it with the docs build's `js-yaml` and make it lint-not-block: hard-fail only on
docs-breakers, warn on consistency issues.

```text
1. Add a minimal Node package for the script's deps:
   - Create .github/scripts/package.json declaring js-yaml as a dependency.
   - Update .github/workflows/validate-frontmatter.yml to install deps in .github/scripts
     before running the validator.

2. Rewrite .github/scripts/validate-frontmatter.js:
   - Parse the HTML-comment front matter with js-yaml (matching sync-ai-cookbook.js).
   - Classify findings into two tiers:
     * HARD ERRORS (exit 1) — would break the docs build: invalid YAML; missing H1 in the
       body; missing `description`.
     * WARNINGS (exit 0, reported) — consistency: missing/empty `tags` or `priority`;
       `tags:[` spacing violation; tag not in the accept-list; wrong tag ordering;
       presence of forbidden `last_updated`/`title` keys.
   - Load the tag accept-list + synonym map from the Step 4 source of truth
     (toolkit/skills/recipe-writing/references/tags.json).
   - Keep the existing top-level-vs-nested README discovery logic.
   - Print a clear per-file report grouped by tier.

3. Write tests for the validator at .github/scripts/test-validate-frontmatter.mjs
   (node:test), covering YOUR logic only:
   - A valid recipe README passes with exit 0.
   - Invalid YAML → hard error.
   - Missing H1 → hard error.
   - `tags:[agents,python]` (no space) → warning, not hard error.
   - A tag outside the accept-list → warning.
   - Presence of `last_updated` → warning.
   - Add an npm `test` script and run it in CI.

4. Reconcile docs: update CONTRIBUTING.md and CLAUDE.md so their "required front matter"
   claims match the validator's actual tiers (description hard-required; tags/priority
   checked as warnings).

5. Verify: run `node validate-frontmatter.js` against the whole repo; confirm it exits 0
   today (existing recipes have description + H1) and prints the expected warnings for the
   known spacing/vocab inconsistencies.
```

---

## Phase 1 — Validation tooling

### Step 7: Scaffold the `recipe-lint` CLI

**NOTE:** Standalone uv-runnable Python CLI (resolved decision), bundled under
`toolkit/tools/recipe-lint/`. Usable locally, by CI (repo path), and by the skill/agent
(via `${CLAUDE_PLUGIN_ROOT}`). This step builds the skeleton — dispatcher, finding model,
report formatter, registry — with no checks yet.

```text
1. Create a uv package at toolkit/tools/recipe-lint/:
   - pyproject.toml: name "recipe-lint", requires-python ">=3.10", a console entry point
     `recipe-lint = recipe_lint.cli:main`, dev deps pytest. Follow the `python` skill's
     pyproject conventions (ruff + mypy config).
   - src/recipe_lint/__init__.py
   - src/recipe_lint/cli.py
   - src/recipe_lint/findings.py
   - src/recipe_lint/dispatch.py
   - tests/

2. Implement recipe_lint/findings.py:
   - A Finding dataclass: severity ("error"|"warning"), code (str), message (str),
     file (str|None), line (int|None).
   - A format_report(findings, recipe_dir) -> str producing a grouped, readable report.
   - An exit-code rule: errors → nonzero, warnings-only → zero (lint-not-block).

3. Implement recipe_lint/dispatch.py:
   - detect_language(recipe_dir) using the directory-name suffix (`_python` → "python";
     extensible for `_typescript`, `_go` later).
   - run_checks(recipe_dir): resolve language, look up the registered check module, run
     its checks, return list[Finding]. Unknown language → single warning, no crash.

4. Implement recipe_lint/cli.py:
   - argparse: `recipe-lint <recipe-dir> [--format text|json]`.
   - Calls dispatch.run_checks, prints report, returns the exit code from the rule above.

5. Tests (toolkit/tools/recipe-lint/tests/test_dispatch.py, test_findings.py):
   - detect_language maps `foo_python` → "python", unknown suffix → None.
   - exit-code rule: any error → nonzero; only warnings → zero; empty → zero.
   - format_report groups by severity and includes file/line when present.

6. Verify: from the package dir, `uv run recipe-lint ../../../agents/guardrails_hard_rules_python`
   runs and prints an (empty) report with exit 0. Run `ruff check` and `mypy` clean.
```

### Step 8: Implement structural / layout / naming / link checks (python module)

```text
1. Create recipe_lint/checks/python/structure.py implementing checks that read files and
   directory structure (regex/path-based, no AST yet):
   - Required files present: pyproject.toml, README.md, worker.py, start_workflow.py.
   - Required dirs present: activities/, workflows/, tests/ (tests MANDATORY → error if
     missing or empty).
   - Package name in pyproject.toml matches `cookbook-{recipe}-python` (warning on mismatch).
   - Task-queue string `{recipe}-task-queue` appears in worker.py and start_workflow.py
     (warning).
   - No stray top-level entry file duplicating start_workflow.py (warning).
   - README front matter present (delegate detail to the JS validator; here just check the
     comment block + H1 exist) — warning if absent.
   - Relative links in README resolve to real paths; referenced _assets/ images exist
     (warning on broken link/image).

2. Register the module in dispatch.py for language "python".

3. Tests (tests/test_structure_checks.py) using small fixture recipe dirs under
   tests/fixtures/ (a "good" recipe and several "bad" variants):
   - Missing tests/ → error finding with the right code.
   - Wrong package name → warning.
   - Broken README relative link → warning.
   - A clean fixture → no findings.

4. Verify: run `recipe-lint` against all 13 real recipes; capture the findings as the
   Phase-3 worklist. Confirm no check crashes on any real recipe.
```

### Step 9: Implement code-convention checks (AST, python module)

```text
1. Create recipe_lint/checks/python/conventions.py using the `ast` module:
   - Flag LLM client constructors (AsyncOpenAI, AsyncAnthropic, etc.) NOT passing
     max_retries=0 (warning).
   - Flag Client.connect(...) calls without data_converter=pydantic_data_converter, and
     WorkflowEnvironment.start_time_skipping(...) without it in tests (warning).
   - Flag workflow.execute_activity(...) calls with no start_to_close_timeout kwarg
     (warning).
   - Flag activity functions whose body calls an LLM/HTTP client but never raises
     ApplicationError(non_retryable=True) in an except (best-effort warning — judgment
     cases deferred to the reviewer agent).
   - Flag stale/hardcoded model-name string literals against a small known-current set
     (warning; the set lives in conventions.py and is easy to update).

2. Register these checks alongside structure checks for language "python".

3. Tests (tests/test_convention_checks.py) using targeted code-snippet fixtures:
   - AsyncAnthropic() without max_retries=0 → warning; with it → clean.
   - Client.connect without the data converter → warning.
   - execute_activity without timeout → warning.
   - A known-stale model literal → warning.

4. Verify: run against all real recipes; confirm findings are accurate (manually validate
   a sample of 3) and there are no false positives that would create noise. Tune patterns
   if needed. ruff + mypy clean.
```

### Step 10: Wire ruff + mypy into recipe-lint

```text
1. Add recipe_lint/checks/python/quality.py:
   - Shell out to `ruff check` and `mypy` against the recipe directory, using a
     toolkit-provided config (ship toolkit/tools/recipe-lint/configs/ruff.toml and
     mypy.ini reflecting the `python` skill's strict defaults).
   - Parse their output into Finding objects (warning severity).
   - Gracefully degrade to a single informational finding if ruff/mypy are not installed.

2. Register quality checks for language "python".

3. Tests (tests/test_quality_checks.py):
   - Given a fixture with an obvious ruff violation, a finding is produced.
   - Given a clean fixture, none.
   - Missing tool → informational finding, no crash.

4. Verify: run full `recipe-lint` (structure + conventions + quality) on the corpus;
   confirm the report is coherent and the exit code follows the lint-not-block rule.
```

### Step 11: Author the Vale ruleset (prose layer)

**NOTE:** Deliberately small — front matter, heading case, banned marketing words, the
`*File:*` convention. Not VPT's 30 rules. Lives under the plugin root so it travels with
the plugin; invoked via `--config`.

```text
1. Create the Vale config and a small ruleset under the toolkit:
   - toolkit/.vale.ini: StylesPath=styles, Vocab=AICookbook, applied to **/README.md.
   - toolkit/styles/AICookbook/ with a minimal set of rules:
     * HeadingsSentenceCase.yml — section headings in sentence case.
     * MarketingLanguage.yml — flag banned marketing/AI-giveaway words (small curated list).
     * FileAnnotation.yml — flag code blocks in a "Create the X" section not preceded by a
       `*File: ...*` line (best-effort existence check).
   - toolkit/styles/config/vocabularies/AICookbook/accept.txt + reject.txt (Temporal
     primitives, provider names, etc.).

2. Keep the rule set intentionally short; document in toolkit/styles/AICookbook/README.md
   what each rule does and the "must earn its place" bar for adding more.

3. Verify: run `vale --config toolkit/.vale.ini agents/ foundations/ deep_research/ mcp/`
   over existing READMEs; confirm the rules fire only on genuine issues. Trim noise.
```

### Step 12: Build the `review-recipe` command + `recipe-reviewer` agent

**NOTE:** Both live under the `toolkit/` plugin root (NOT `.claude/`, which is project-local
and not a plugin component path). Use the current agent description style — prose triggers +
a "When to invoke" body section — not XML `<example>` blocks.

```text
1. Author toolkit/agents/recipe-reviewer.md:
   - Frontmatter: `name: recipe-reviewer`; `description` in prose
     ("Use this agent when… Typical triggers include reviewing a recipe before a PR,
     checking a recipe against the cookbook style, validating a generated recipe. See
     'When to invoke' in the agent body."); `model: inherit`; `color`; `tools` limited to
     ["Read", "Grep", "Glob", "Bash"].
   - Body (second person): a "When to invoke" section with 2–4 prose scenarios; then the
     process — read the four references at ${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/
     references/ first; run
     `uv run --project ${CLAUDE_PLUGIN_ROOT}/tools/recipe-lint recipe-lint <dir>`,
     `vale --config ${CLAUDE_PLUGIN_ROOT}/.vale.ini <dir>/README.md`, and the recipe's
     tests (`uv run pytest tests/`); then the judgment-only checks the linter can't
     (code-sandwich completeness, section purpose, prose clarity, context-dependent
     Temporal capitalization). Output a structured report grouped error/warning/suggestion.
     State that tooling is advisory and the agent is the enforcer.

2. Author toolkit/commands/review-recipe.md:
   - Frontmatter: `description`; `allowed-tools: Read, Grep, Glob, Bash, Task`;
     `argument-hint: <recipe-dir>`.
   - Body (instructions FOR Claude): resolve the target recipe dir; launch the
     recipe-reviewer agent against it; combine recipe-lint + Vale + test results + agent
     findings into one report with a prioritized action list.

3. Verify: load with `--plugin-dir <repo>/toolkit`; run
   `/review-recipe agents/guardrails_hard_rules_python` end to end; confirm the report is
   accurate and actionable, and that `${CLAUDE_PLUGIN_ROOT}` paths resolve.
```

### Step 13: Wire the toolkit into CI (non-blocking lint + report)

```text
1. Add .github/workflows/lint-recipes.yml:
   - On pull_request touching recipe dirs, detect changed recipe directories (reuse the
     detection approach from validate-python-projects.yml).
   - For each changed recipe: run recipe-lint by repo path
     (`uv run --project toolkit/tools/recipe-lint recipe-lint <dir>`) and
     `vale --config toolkit/.vale.ini <dir>/README.md`.
   - Report findings as job summary output. Do NOT fail the job on warnings
     (lint-not-block). Allow failure only if recipe-lint emits an error-severity finding
     (e.g. missing mandatory tests/).

2. Confirm validate-frontmatter.yml (Step 6) remains the hard-gate for docs-breakers, and
   validate-python-projects.yml remains the hard-gate for tests. The new workflow is purely
   advisory for style/consistency.

3. Verify: open a draft PR with a deliberately non-conforming recipe; confirm CI reports
   the issues, does not block on warnings, and blocks on a missing-tests error.
```

---

## Phase 2 — Reconcile generation + packaging

### Step 14: Move + point `recipe-ify` at the single source of truth

**NOTE:** Relocates Angie's `.claude/commands/recipe-ify.md` to `toolkit/commands/recipe-ify.md`
and refactors it. Coordinate with Angie per the branch-flow constraint before merging back.

```text
1. Move .claude/commands/recipe-ify.md → toolkit/commands/recipe-ify.md and refactor:
   - Add command frontmatter (description, argument-hint, allowed-tools).
   - Replace the inline convention prose with instructions to READ
     ${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/ (structure, layout,
     frontmatter, code-conventions) and follow them as the authoritative source.
   - Change the README output spec to generate the CANONICAL walkthrough README (Step 2),
     not the brief style.
   - Keep the audience framing and the file-generation flow.
   - After generation, instruct it to run recipe-lint on the new recipe and fix findings.

2. Verify: load via --plugin-dir; run `/recipe-ify` on a simple pattern description;
   confirm the generated recipe passes recipe-lint and the README matches the canonical
   structure.
```

### Step 15: Move + point `recipe-scout` at the SSOT + wishlist

**NOTE:** Relocates Angie's `.claude/commands/recipe-scout.md` to `toolkit/commands/`.
Coordinate with Angie.

```text
1. Move .claude/commands/recipe-scout.md → toolkit/commands/recipe-scout.md and refactor:
   - Add command frontmatter.
   - Reference ${CLAUDE_PLUGIN_ROOT}/skills/recipe-writing/references/ for what a "good
     recipe" is, instead of restating conventions inline.
   - Keep the AI-building-block taxonomy and the coverage wishlist (scout-specific; no
     better home yet).
   - Ensure proposal cards reference the canonical structure for "how it would be structured".

2. Verify: run `/recipe-scout` against a sample external repo; confirm proposal cards are
   coherent and reference the canonical conventions.
```

### Step 16: Add `new-recipe` command + recipe template

```text
1. Create toolkit/templates/recipe-skeleton/ — a minimal, runnable recipe matching
   layout.md and the canonical README (placeholders in ALL_CAPS): pyproject.toml,
   README.md, worker.py, start_workflow.py, activities/, workflows/, tests/.

2. Author toolkit/commands/new-recipe.md:
   - Frontmatter: description; allowed-tools (Read, Write, Bash, AskUserQuestion, Glob);
     argument-hint "<category/recipe-name>".
   - Body (FOR Claude): ask for recipe name, category, language, provider; copy the
     skeleton from ${CLAUDE_PLUGIN_ROOT}/templates/recipe-skeleton/ into the target
     category dir at the repo root; fill placeholders (package name, task queue, front
     matter); remind the author what to fill in and to run recipe-lint.

3. Verify: `/new-recipe foundations/example_python` produces a skeleton at the repo root
   that passes recipe-lint (modulo intentional TODO placeholders) and `uv sync` succeeds.
```

### Step 17: Package the `toolkit/` plugin + local-load docs

**NOTE:** Local, `--plugin-dir`-loaded plugin — no marketplace. The `agents/` collision is
already avoided by rooting in `toolkit/` (the recipe `agents/` category is outside the
plugin tree). This step just finalizes the manifest and the load instructions.

```text
1. Create toolkit/.claude-plugin/plugin.json: name "ai-cookbook-toolkit", version,
   description, author, keywords. No custom component paths needed — commands/, agents/,
   skills/ sit at the toolkit (plugin) root and auto-discover.

2. Confirm every component is under toolkit/ and uses ${CLAUDE_PLUGIN_ROOT} for intra-
   plugin references (skill body, commands, agent). Confirm nothing the plugin needs lives
   under .claude/ or relies on repo-relative paths.

3. Write load + usage instructions:
   - In the root README.md, document the repo's THREE purposes (recipe content; the
     Vale + recipe-lint consistency tooling; the local Claude Code plugin) and how to load
     it: `claude --plugin-dir <path-to-repo>/toolkit`.
   - Note that local `.claude/` skills remain fine for one-off use, but `toolkit/` is the
     durable home.

4. Verify: in a fresh session, `claude --plugin-dir <repo>/toolkit` exposes the skill and
   all four commands; `/help` lists them; the content `agents/` recipes are unaffected
   (no recipe README is mis-loaded as an agent).
```

---

## Phase 3 — Backfill consistency

### Step 18: Backfill front matter across all recipes (no renames)

```text
1. Using the Step 4 worklist, edit each recipe README's front matter to the schema:
   fix `tags:[` spacing, map tags to the accept-list + ordering, normalize priority to the
   documented bands. Do NOT rename any directory.

2. After each edit, run `node .github/scripts/validate-frontmatter.js` for that file.

3. Verify: validator reports zero warnings across the corpus (or only intentional,
   documented exceptions).
```

### Step 19: Converge README structure to canonical (outliers)

```text
1. For each recipe whose README is NOT the canonical walkthrough (per Step 2's deviation
   list), restructure it to the walkthrough shape. EXCLUDE the open-item recipes:
   - guardrails_hard_rules_python (Angie's — coordinate separately).
   - human_in_the_loop_python (its own PR + rationale, per Open items).

2. After each, run `vale --config toolkit/.vale.ini <dir>/README.md` and
   `uv run --project toolkit/tools/recipe-lint recipe-lint <dir>`.

3. Verify: every converged README passes Vale's structure-related rules and renders
   correctly (spot-check the H1 + "Create the X" sections).
```

### Step 20: Backfill code quality across recipes

```text
1. For each recipe, run recipe-lint and apply the quality bar: ruff format + fix, resolve
   mypy strict findings, address convention warnings (max_retries, data converter,
   timeouts, ApplicationError, model names). Keep changes minimal — convergence, not rewrite.

2. Ensure each recipe still passes `uv run pytest tests/ --timeout=30`.

3. Verify: recipe-lint reports no error-severity findings across the corpus and only
   acknowledged warnings.
```

### Step 21: Final integration

```text
1. Update CONTRIBUTING.md and CLAUDE.md to describe the toolkit: the skill, the commands,
   recipe-lint, Vale, the (non-blocking) CI lint workflow, and the
   `claude --plugin-dir <repo>/toolkit` load step. Point contributors at `/new-recipe` and
   `/review-recipe`.

2. Run the full toolchain end to end on a fresh clone:
   - `node .github/scripts/validate-frontmatter.js` → exit 0, no warnings.
   - recipe-lint on every recipe → no errors.
   - `vale --config toolkit/.vale.ini` on every README → clean.
   - All recipe test suites pass.

3. Verify: CI is green on the branch; the plugin loads via --plugin-dir; a dry-run
   `/new-recipe` → `/review-recipe` cycle works on a throwaway recipe.
```

---

## Implementation Guidelines

- **No TDD ceremony.** Build, then verify by running the tooling against the real recipe
  corpus. The validator (Step 6) and `recipe-lint` (Steps 7–10) ship with real unit tests
  because they have non-trivial logic; everything else is verified by execution.
- **Single source of truth.** Conventions live once in
  `toolkit/skills/recipe-writing/references/`. The validator, linter, commands, and agent
  reference them — never restate them.
- **Lint, don't block.** CI reports style/consistency; it hard-fails only on docs-breakers
  (invalid YAML, missing H1) and missing mandatory tests. The skill/agent is the enforcer.
- **No renames.** Directory names are public URLs; Phase 3 changes content, never paths.
- **Respect the branch flow.** Steps 1–13 and 16–21 add or edit toolkit/content files.
  Steps 14–15 move and modify Angie's command files — coordinate before merging back.
- **Earn every rule.** Adding a Vale or recipe-lint check requires a real inconsistency it
  catches. Prefer few, high-value checks.
- **Language-extensible.** All Python-specific logic lives under
  `toolkit/tools/recipe-lint/.../checks/python/`; adding a language is a new module behind
  the dispatcher, not a rework.
- **Match surrounding style.** Follow the `python` skill for all Python; match existing
  recipe idioms when editing recipes.

### Plugin mechanics (applies to every plugin step)

- The plugin is **rooted in `toolkit/`** and **loaded locally via
  `claude --plugin-dir <repo>/toolkit`** — not published. This keeps the recipe `agents/`
  category outside the plugin's component tree, avoiding auto-discovery collisions.
- Auto-discovered component dirs (`commands/`, `agents/`, `skills/`) sit at the **plugin
  root** (`toolkit/`). `.claude/` is project-local and is NOT a plugin component path;
  local `.claude/` skills are acceptable for one-offs only.
- Custom paths in `plugin.json` **supplement** defaults, they do not replace them — so we
  relocate components rather than try to redirect a default directory.
- Plugin components reference their own files with **`${CLAUDE_PLUGIN_ROOT}/…`** (resolves
  to the `--plugin-dir` target). CI and contributors invoke the same tools by repo path
  (`toolkit/…`). No hardcoded absolute paths.
- Agent files use the current style: **prose triggers in `description` + a "When to invoke"
  body section** (not XML `<example>` blocks); required frontmatter `name` / `description`
  / `model: inherit` / `color`, with least-privilege `tools`.

## Success Metrics

- One source of truth exists (`toolkit/skills/recipe-writing/references/`) and every tool
  references it; no convention is duplicated in command prose.
- `recipe-lint` runs standalone via `uv` and reports accurate findings on all 13 recipes
  with no crashes and no noisy false positives.
- The front-matter validator parses with `js-yaml`, matches the docs build, and tiers
  hard-errors vs warnings correctly.
- Vale runs on every README with a small, high-signal rule set via `--config toolkit/.vale.ini`.
- `/new-recipe` → `/review-recipe` is a working author loop; `/recipe-ify` generates
  canonical, lint-clean recipes.
- CI reports consistency non-blockingly and hard-gates only docs-breakers and tests.
- The `toolkit/` plugin loads via `--plugin-dir` without disturbing the `agents/` recipe
  category, and all intra-plugin references resolve through `${CLAUDE_PLUGIN_ROOT}`.
- After Phase 3: the corpus is consistent — front matter clean, READMEs canonical (except
  the two coordinated open-item recipes), code meets the quality bar — with zero renames.
