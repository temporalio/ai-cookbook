// ABOUTME: Unit tests for the recipe front-matter validator's evaluate() logic.
// Covers the tier split (hard errors vs warnings) for our own rules, not js-yaml itself.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate, extractFrontMatter, hasH1 } from './validate-frontmatter.js';

// A self-contained tag vocabulary mirroring tags.json, so tests are hermetic.
const TAGS = {
  order: ['category', 'language', 'provider'],
  categories: ['agents', 'foundations', 'deep_research', 'mcp'],
  languages: ['python'],
  providers: ['openai', 'anthropic', 'litellm'],
  synonyms: { claude: 'anthropic', 'provider-neutral': 'litellm' },
};

const valid = `<!--
description: Demonstrates a thing.
tags: [foundations, python, openai]
priority: 500
-->

# Hello World

Body text.
`;

test('a valid recipe README passes with no errors or warnings', () => {
  const { errors, warnings } = evaluate(valid, TAGS);
  assert.deepEqual(errors, []);
  assert.deepEqual(warnings, []);
});

test('invalid YAML is a hard error', () => {
  const content = `<!--
description: "unterminated
tags: [foundations, python]
-->

# Title
`;
  const { errors } = evaluate(content, TAGS);
  assert.ok(errors.some((e) => e.includes('invalid YAML')), `expected YAML error, got ${JSON.stringify(errors)}`);
});

test('missing H1 is a hard error', () => {
  const content = `<!--
description: Demonstrates a thing.
tags: [foundations, python]
priority: 500
-->

No heading here, just prose.
`;
  const { errors } = evaluate(content, TAGS);
  assert.ok(errors.some((e) => e.includes('missing H1')), `expected H1 error, got ${JSON.stringify(errors)}`);
});

test('missing description is a hard error', () => {
  const content = `<!--
tags: [foundations, python]
priority: 500
-->

# Title
`;
  const { errors } = evaluate(content, TAGS);
  assert.ok(errors.some((e) => e.includes('description')));
});

test('a level-2 heading does not satisfy the H1 requirement', () => {
  assert.equal(hasH1('## Not an H1\n'), false);
  assert.equal(hasH1('# Real H1\n'), true);
});

test('tags:[ with no space is a warning, not a hard error', () => {
  const content = `<!--
description: Demonstrates a thing.
tags:[foundations, python]
priority: 500
-->

# Title
`;
  const { errors, warnings } = evaluate(content, TAGS);
  assert.deepEqual(errors, []);
  assert.ok(warnings.some((w) => w.includes('space after the colon')));
});

test('a tag outside the accept-list is a warning', () => {
  const content = `<!--
description: Demonstrates a thing.
tags: [foundations, python, s3]
priority: 500
-->

# Title
`;
  const { errors, warnings } = evaluate(content, TAGS);
  assert.deepEqual(errors, []);
  assert.ok(warnings.some((w) => w.includes("unknown tag 's3'")));
});

test('a synonym tag is a warning pointing at the canonical form', () => {
  const content = `<!--
description: Demonstrates a thing.
tags: [agents, python, claude]
priority: 500
-->

# Title
`;
  const { warnings } = evaluate(content, TAGS);
  assert.ok(warnings.some((w) => w.includes("'claude' should be 'anthropic'")));
});

test('tags out of order is a warning', () => {
  const content = `<!--
description: Demonstrates a thing.
tags: [python, foundations]
priority: 500
-->

# Title
`;
  const { warnings } = evaluate(content, TAGS);
  assert.ok(warnings.some((w) => w.includes('out of order')));
});

test('a forbidden last_updated key is a warning', () => {
  const content = `<!--
description: Demonstrates a thing.
tags: [foundations, python]
priority: 500
last_updated: 2026-01-01
-->

# Title
`;
  const { errors, warnings } = evaluate(content, TAGS);
  assert.deepEqual(errors, []);
  assert.ok(warnings.some((w) => w.includes("remove 'last_updated'")));
});

test('a non-integer priority is a warning', () => {
  const content = `<!--
description: Demonstrates a thing.
tags: [foundations, python]
priority: high
-->

# Title
`;
  const { warnings } = evaluate(content, TAGS);
  assert.ok(warnings.some((w) => w.includes('priority must be an integer')));
});

test('missing front matter comment is a hard error', () => {
  const { errors } = evaluate('# Title\n\nNo comment.\n', TAGS);
  assert.ok(errors.some((e) => e.includes('missing front matter comment')));
});

test('extractFrontMatter joins a wrapped multi-line description (docs-sync parity)', () => {
  const content = `<!--
description: Build a simple deep research system embodying the standard
deep research architecture.
tags: [deep_research, python]
priority: 399
-->

# Deep Research
`;
  const fm = extractFrontMatter(content);
  assert.ok(!('error' in fm), `unexpected error: ${JSON.stringify(fm)}`);
  assert.match(fm.data.description, /standard deep research architecture\.$/);
});
