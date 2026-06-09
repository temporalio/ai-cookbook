# AICookbook Vale style

The prose layer of the cookbook toolkit. Deliberately **minimal** — this is a cookbook,
not the Validated Patterns program, so it does not carry VP's ~30 rules. Run it with:

```bash
vale --config cookbook-toolkit/.vale.ini <recipe>/README.md
```

## Rules

| Rule | Level | What it catches |
| :--- | :--- | :--- |
| `MarketingLanguage` | warning | Marketing / AI-giveaway words (`leverage`, `seamless`, `delve`, `dive into`, `cutting-edge`, …). Recipes say plainly what the code does. |

## The "earn its place" bar

Add a rule only when there is a **real, recurring inconsistency** in the corpus that the
rule catches with low false-positives. Two candidates were evaluated and **rejected**:

- **Heading sentence-case** — rejected. The corpus consistently uses Title Case headings
  ("Create the Workflow Starter", "Application Components"). A sentence-case rule fired 49
  times against the de-facto convention — noise, not signal.
- **`*File:*` / code-sandwich annotation** — not a Vale rule. "A code block must be
  preceded by a `*File: path*` line" is a structural relationship Vale can't express
  cleanly; it's left to the `recipe-reviewer` agent's judgment-level review.

When a new rule is proposed, run it across all recipe READMEs first (as above) and keep it
only if its hits are genuine.
