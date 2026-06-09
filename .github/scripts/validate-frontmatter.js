// ABOUTME: Validates HTML-comment front matter in AI cookbook recipe README files.
// Parses with js-yaml (parity with the docs build) and tiers findings: hard errors
// (docs-breakers) exit non-zero; consistency warnings are reported but do not block.

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'js-yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Repo root: this script lives at <repo>/.github/scripts/.
const REPO_ROOT = process.argv[2] || path.resolve(__dirname, '..', '..');
const TAGS_PATH = path.join(REPO_ROOT, 'toolkit', 'skills', 'recipe-writing', 'references', 'tags.json');

const FORBIDDEN_KEYS = ['last_updated', 'title'];

/**
 * Replicate the docs-sync normalization (bin/sync-ai-cookbook.js) so this validator
 * parses front matter identically to the published build, then yaml.load it.
 * @returns {{ data: object } | { error: string }}
 */
function parseCommentBody(commentBody) {
  const normalizedLines = [];
  for (const rawLine of commentBody.split('\n')) {
    const line = rawLine.trimEnd();
    const isKeyLine = /^[ \t]*[\w-]+:/.test(line);
    if (isKeyLine || line.trim().length === 0 || normalizedLines.length === 0) {
      normalizedLines.push(line);
    } else {
      normalizedLines[normalizedLines.length - 1] = `${normalizedLines[normalizedLines.length - 1]} ${line.trim()}`;
    }
  }
  const normalized = normalizedLines
    .join('\n')
    .replace(/(^|\n)([ \t]*[\w-]+):(?=\S)/g, (full, prefix, key) => `${prefix}${key}: `);
  try {
    const data = yaml.load(normalized) ?? {};
    if (data === null || typeof data !== 'object' || Array.isArray(data)) {
      return { error: 'front matter must be a YAML object' };
    }
    return { data };
  } catch (err) {
    return { error: `invalid YAML: ${err.message}` };
  }
}

/**
 * Extract the leading HTML-comment front matter and the body that follows it.
 * @returns {{ raw: string, data: object, body: string } | { error: string }}
 */
export function extractFrontMatter(content) {
  const match = /^\s*<!--([\s\S]*?)-->/.exec(content);
  if (!match) {
    return { error: 'missing front matter comment' };
  }
  const raw = match[1].replace(/\r/g, '');
  const parsed = parseCommentBody(raw.trim());
  if ('error' in parsed) {
    return { error: parsed.error };
  }
  const body = content.slice(match.index + match[0].length).replace(/^\s+/, '');
  return { raw, data: parsed.data, body };
}

/** The body has a level-1 H1 as its first heading. */
export function hasH1(body) {
  return /^#\s+\S/.test(body.trimStart());
}

/** Map a (canonicalized) tag to its axis index, or null if unknown. */
function tagAxis(tag, tagsConfig) {
  if (tagsConfig.categories.includes(tag)) return 0;
  if (tagsConfig.languages.includes(tag)) return 1;
  if (tagsConfig.providers.includes(tag)) return 2;
  return null;
}

/**
 * Evaluate one README's content against the schema.
 * @returns {{ errors: string[], warnings: string[] }}
 *   errors are docs-breakers (caller exits non-zero); warnings are consistency issues.
 */
export function evaluate(content, tagsConfig) {
  const errors = [];
  const warnings = [];

  const fm = extractFrontMatter(content);
  if ('error' in fm) {
    errors.push(fm.error);
    return { errors, warnings };
  }
  const { raw, data, body } = fm;

  // --- Hard errors: would break the docs build ---
  if (!hasH1(body)) {
    errors.push('missing H1 title (the docs build promotes the H1 to the page title)');
  }
  if (!data.description || String(data.description).trim() === '') {
    errors.push('missing required field: description');
  }

  // --- Warnings: consistency ---
  // Raw spacing: `tags:[` with no space after the colon.
  if (/(^|\n)[ \t]*tags:\[/.test(raw)) {
    warnings.push('tags: needs a space after the colon (use `tags: [ ... ]`)');
  }

  const tags = data.tags;
  if (tags === undefined || (Array.isArray(tags) && tags.length === 0)) {
    warnings.push('missing or empty tags');
  } else if (!Array.isArray(tags)) {
    warnings.push('tags must be a list');
  } else {
    const axes = [];
    for (const rawTag of tags) {
      const tag = String(rawTag).trim();
      if (Object.prototype.hasOwnProperty.call(tagsConfig.synonyms, tag)) {
        warnings.push(`tag '${tag}' should be '${tagsConfig.synonyms[tag]}'`);
        axes.push(tagAxis(tagsConfig.synonyms[tag], tagsConfig));
        continue;
      }
      const axis = tagAxis(tag, tagsConfig);
      if (axis === null) {
        warnings.push(`unknown tag '${tag}' (not in the accept-list: category, language, or provider)`);
      }
      axes.push(axis);
    }
    // Ordering: known-axis tags must be non-decreasing (category → language → provider).
    const known = axes.filter((a) => a !== null);
    for (let i = 1; i < known.length; i++) {
      if (known[i] < known[i - 1]) {
        warnings.push('tags out of order (use category, then language, then provider)');
        break;
      }
    }
  }

  if (data.priority === undefined) {
    warnings.push('missing priority');
  } else if (!Number.isInteger(data.priority)) {
    warnings.push('priority must be an integer');
  }

  for (const key of FORBIDDEN_KEYS) {
    if (Object.prototype.hasOwnProperty.call(data, key)) {
      warnings.push(`remove '${key}' from front matter (derived by the docs sync)`);
    }
  }

  return { errors, warnings };
}

// ---------------------------------------------------------------------------
// File discovery (only top-level recipe READMEs are validated; nested READMEs
// inside a recipe are exempt). Ported from the original validator.
// ---------------------------------------------------------------------------

async function findReadmeFiles(rootDir) {
  const stack = [rootDir];
  const readmes = [];
  while (stack.length > 0) {
    const current = stack.pop();
    let entries;
    try {
      entries = await fs.readdir(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (entry.name === '.git' || entry.name === '.github' || entry.name === 'node_modules') continue;
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(entryPath);
      } else if (entry.isFile() && entry.name.toLowerCase() === 'readme.md') {
        readmes.push(entryPath);
      }
    }
  }
  return readmes;
}

async function hasParentReadme(readmePath, root) {
  let dir = path.dirname(readmePath);
  const normalizedRoot = path.resolve(root);
  while (dir !== normalizedRoot && dir !== path.dirname(dir)) {
    dir = path.dirname(dir);
    if (dir === normalizedRoot || dir.length < normalizedRoot.length) break;
    try {
      await fs.access(path.join(dir, 'README.md'));
      return true;
    } catch {
      /* keep walking */
    }
  }
  return false;
}

async function loadTagsConfig() {
  const text = await fs.readFile(TAGS_PATH, 'utf8');
  return JSON.parse(text);
}

async function main() {
  console.log(`[validate-frontmatter] repo root: ${REPO_ROOT}\n`);

  let tagsConfig;
  try {
    tagsConfig = await loadTagsConfig();
  } catch (err) {
    console.error(`Error: cannot load tag vocabulary at ${TAGS_PATH}: ${err.message}`);
    process.exit(1);
  }

  const allReadmes = await findReadmeFiles(REPO_ROOT);
  const recipeReadmes = [];
  for (const readme of allReadmes) {
    if (path.dirname(readme) === path.resolve(REPO_ROOT)) continue; // skip root README
    if (path.resolve(readme).includes(`${path.sep}toolkit${path.sep}`)) continue; // skip toolkit docs
    if (!(await hasParentReadme(readme, REPO_ROOT))) recipeReadmes.push(readme);
  }

  let errorCount = 0;
  let warningCount = 0;

  for (const readmePath of recipeReadmes.sort()) {
    const rel = path.relative(REPO_ROOT, readmePath);
    const content = await fs.readFile(readmePath, 'utf8');
    const { errors, warnings } = evaluate(content, tagsConfig);
    errorCount += errors.length;
    warningCount += warnings.length;

    if (errors.length === 0 && warnings.length === 0) {
      console.log(`✓ ${rel}`);
      continue;
    }
    console.log(`• ${rel}`);
    for (const e of errors) console.log(`    ✗ error:   ${e}`);
    for (const w of warnings) console.log(`    ⚠ warning: ${w}`);
  }

  console.log('');
  console.log(`Checked ${recipeReadmes.length} recipe README(s): ${errorCount} error(s), ${warningCount} warning(s).`);

  if (errorCount > 0) {
    console.error('\n❌ Hard errors would break the docs build. Fix them.');
    process.exit(1);
  }
  console.log('\n✅ No docs-breaking errors. Warnings (if any) are consistency issues to address.');
  process.exit(0);
}

// Run only when invoked directly (not when imported by tests).
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error('Unexpected error:', err);
    process.exit(1);
  });
}
