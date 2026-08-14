#!/usr/bin/env node
'use strict';

/**
 * Keeps code snippets embedded in recipe README.md files in sync with the
 * actual recipe source code.
 *
 * README-side markers (HTML comments, following the naming used by Temporal's
 * own Snipsync tool in temporalio/documentation, scoped down for a same-repo
 * README+source layout):
 *
 *   Whole file, no source changes needed:
 *     <!--SNIPSTART:file activities/classify.py-->
 *     <!--SNIPEND-->
 *
 *   Fragment located by a unique regex boundary, no source changes needed:
 *     <!--SNIPSTART activities/classify.py {"startPattern": "...", "endPattern": "..."}-->
 *     <!--SNIPEND-->
 *
 *   Fragment located by an explicit marker in source (fallback for when no
 *   stable regex boundary exists):
 *     <!--SNIPSTART workflows/human_in_the_loop_workflow.py:signal-handler-->
 *     <!--SNIPEND-->
 *   ...matched in source by:
 *     # @@@SNIPSTART signal-handler
 *     ...
 *     # @@@SNIPEND
 *
 *   Optional "selectedLines" (1-indexed, relative to the extracted region,
 *   inclusive ranges) renders only some lines of an otherwise-larger match,
 *   with "..." auto-inserted at gaps:
 *     <!--SNIPSTART activities/classify.py {"startPattern": "...", "endPattern": "...", "selectedLines": ["1", "6-7"]}-->
 *
 * The tool always writes the fence itself (backticks + language tag inferred
 * from the source file's extension) between the two marker comments — README
 * authors never hand-write the fence.
 *
 * Usage:
 *   node sync-readme-snippets.js --check [path]
 *   node sync-readme-snippets.js --fix   [path]
 */

const fs = require('fs');
const path = require('path');

const SKIP_DIR_NAMES = new Set([
  '.git',
  '.github',
  '.venv',
  'node_modules',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  'dist',
  'build',
]);

const LANG_BY_EXT = {
  '.py': 'python',
  '.ts': 'typescript',
  '.tsx': 'tsx',
  '.js': 'javascript',
  '.jsx': 'jsx',
  '.go': 'go',
};

const OPEN_MARKER_RE =
  /^[ \t]*<!--SNIPSTART(:file)?[ \t]+(\S+?)(?::([a-z0-9][a-z0-9-]*))?(?:[ \t]+(\{.*\}))?[ \t]*-->[ \t]*$/;
const CLOSE_MARKER_RE = /^[ \t]*<!--SNIPEND-->[ \t]*$/;
const SOURCE_MARKER_RE = /^[ \t]*(?:#|\/\/)[ \t]*@@@SNIP(START|END)(?:[ \t]+([a-z0-9][a-z0-9-]*))?[ \t]*$/;

class SnippetError extends Error {}

// ---------------------------------------------------------------------------
// Discovery (mirrors validate-frontmatter.js's walk/hasParentReadme logic)
// ---------------------------------------------------------------------------

function findReadmeFiles(rootDir) {
  const stack = [rootDir];
  const readmes = [];

  while (stack.length > 0) {
    const current = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (SKIP_DIR_NAMES.has(entry.name)) continue;

      const entryPath = path.join(current, entry.name);

      if (entry.isDirectory()) {
        stack.push(entryPath);
        continue;
      }

      if (entry.isFile() && entry.name.toLowerCase() === 'readme.md') {
        readmes.push(entryPath);
      }
    }
  }

  return readmes;
}

function hasParentReadme(readmePath, cookbookRoot) {
  const normalizedRoot = path.resolve(cookbookRoot);
  let currentDir = path.dirname(readmePath);

  while (currentDir !== normalizedRoot && currentDir !== path.dirname(currentDir)) {
    currentDir = path.dirname(currentDir);
    if (currentDir === normalizedRoot) break;
    if (currentDir.length < normalizedRoot.length) break;

    const parentReadme = path.join(currentDir, 'README.md');
    if (fs.existsSync(parentReadme)) return true;
  }

  return false;
}

function findTopLevelReadmes(cookbookDir) {
  const allReadmes = findReadmeFiles(cookbookDir);
  const resolvedRoot = path.resolve(cookbookDir);

  return allReadmes.filter((readme) => {
    if (path.dirname(readme) === resolvedRoot) return false; // repo-root README
    return !hasParentReadme(readme, cookbookDir);
  });
}

// ---------------------------------------------------------------------------
// README-side marker parsing
// ---------------------------------------------------------------------------

/**
 * Splits text into lines while recording each line's start offset, so marker
 * matches can be mapped back to byte ranges in the original content.
 */
function splitWithOffsets(text) {
  const lines = [];
  let offset = 0;
  for (const line of text.split('\n')) {
    lines.push({ text: line, start: offset });
    offset += line.length + 1; // +1 for the '\n' this split ate
  }
  return lines;
}

/**
 * Finds every SNIPSTART/SNIPEND marker pair in a README's content.
 * Returns entries with the byte ranges needed to splice in a fix, plus the
 * current raw content between the markers (for --check comparison).
 */
function parseReadmeMarkers(content) {
  const lines = splitWithOffsets(content);
  const markers = [];
  let i = 0;

  while (i < lines.length) {
    const openMatch = OPEN_MARKER_RE.exec(lines[i].text);
    if (!openMatch) {
      i++;
      continue;
    }

    const openLineEnd = lines[i].start + lines[i].text.length + 1; // just past this line's \n
    const isFile = Boolean(openMatch[1]);
    const snipPath = openMatch[2];
    const id = openMatch[3] || null;
    let config = {};
    if (openMatch[4]) {
      try {
        config = JSON.parse(openMatch[4]);
      } catch (e) {
        throw new SnippetError(`malformed JSON config on SNIPSTART line ${i + 1}: ${e.message}`);
      }
    }

    let j = i + 1;
    let closeIdx = -1;
    while (j < lines.length) {
      if (CLOSE_MARKER_RE.test(lines[j].text)) {
        closeIdx = j;
        break;
      }
      if (OPEN_MARKER_RE.test(lines[j].text)) {
        throw new SnippetError(
          `nested/unclosed SNIPSTART at line ${i + 1} (found another SNIPSTART at line ${j + 1} before a SNIPEND)`
        );
      }
      j++;
    }

    if (closeIdx === -1) {
      throw new SnippetError(`SNIPSTART at line ${i + 1} has no matching SNIPEND`);
    }

    const closeLineStart = lines[closeIdx].start;
    const currentInner = content.slice(openLineEnd, closeLineStart);

    markers.push({
      openLineNumber: i + 1,
      isFile,
      snipPath,
      id,
      config,
      openLineEnd,
      closeLineStart,
      currentInner,
    });

    i = closeIdx + 1;
  }

  return markers;
}

// ---------------------------------------------------------------------------
// Source extraction
// ---------------------------------------------------------------------------

function langFromExt(filePath) {
  return LANG_BY_EXT[path.extname(filePath)] || '';
}

function dedent(lines) {
  let minIndent = Infinity;
  for (const line of lines) {
    if (line.trim() === '') continue;
    const match = /^[ \t]*/.exec(line);
    minIndent = Math.min(minIndent, match[0].length);
  }
  if (!Number.isFinite(minIndent) || minIndent === 0) return lines;
  return lines.map((line) => (line.trim() === '' ? '' : line.slice(minIndent)));
}

function parseSelectedLines(spec, totalLines) {
  const kept = new Set();
  for (const entry of spec) {
    const rangeMatch = /^(\d+)-(\d+)$/.exec(String(entry).trim());
    if (rangeMatch) {
      const start = Number(rangeMatch[1]);
      const end = Number(rangeMatch[2]);
      for (let n = start; n <= end; n++) kept.add(n);
    } else {
      kept.add(Number(String(entry).trim()));
    }
  }
  for (const n of kept) {
    if (n < 1 || n > totalLines) {
      throw new SnippetError(`selectedLines entry ${n} is out of range (snippet has ${totalLines} lines)`);
    }
  }
  return kept;
}

function applySelectedLines(lines, selectedLinesSpec) {
  const kept = parseSelectedLines(selectedLinesSpec, lines.length);
  const sortedKept = [...kept].sort((a, b) => a - b);
  const result = [];
  let previousKeptIndex = null;

  if (sortedKept.length > 0 && sortedKept[0] !== 1) {
    const indentMatch = /^[ \t]*/.exec(lines[sortedKept[0] - 1]);
    result.push(`${indentMatch[0]}...`);
  }

  for (let idx = 0; idx < lines.length; idx++) {
    const lineNumber = idx + 1;
    if (!kept.has(lineNumber)) continue;

    if (previousKeptIndex !== null && lineNumber !== previousKeptIndex + 1) {
      const indentMatch = /^[ \t]*/.exec(lines[idx]);
      result.push(`${indentMatch[0]}...`);
    }

    result.push(lines[idx]);
    previousKeptIndex = lineNumber;
  }

  if (sortedKept.length > 0 && sortedKept[sortedKept.length - 1] !== lines.length) {
    const indentMatch = /^[ \t]*/.exec(lines[sortedKept[sortedKept.length - 1] - 1]);
    result.push(`${indentMatch[0]}...`);
  }

  return result;
}

function readSourceLines(absPath) {
  if (!fs.existsSync(absPath)) {
    throw new SnippetError(`source file not found: ${absPath}`);
  }
  const raw = fs.readFileSync(absPath, 'utf8').replace(/\r\n/g, '\n');
  return raw.replace(/\n$/, '').split('\n');
}

function extractByFile(absPath) {
  return { lines: readSourceLines(absPath), lang: langFromExt(absPath) };
}

function extractById(absPath, id) {
  const lines = readSourceLines(absPath);
  let startIdx = -1;
  let endIdx = -1;

  for (let idx = 0; idx < lines.length; idx++) {
    const m = SOURCE_MARKER_RE.exec(lines[idx]);
    if (!m) continue;
    if (m[1] === 'START' && m[2] === id) {
      if (startIdx !== -1) {
        throw new SnippetError(`duplicate @@@SNIPSTART ${id} in ${absPath} (line ${idx + 1})`);
      }
      startIdx = idx;
    } else if (m[1] === 'END' && startIdx !== -1 && endIdx === -1 && idx > startIdx) {
      // First SNIPEND after our start closes it. Ids on SNIPEND are optional
      // and not required to match — a file's markers are processed in order.
      if (m[2] && m[2] !== id) continue;
      endIdx = idx;
    }
  }

  if (startIdx === -1) {
    throw new SnippetError(`no @@@SNIPSTART ${id} found in ${absPath}`);
  }
  if (endIdx === -1) {
    throw new SnippetError(`@@@SNIPSTART ${id} in ${absPath} has no matching @@@SNIPEND`);
  }

  return { lines: dedent(lines.slice(startIdx + 1, endIdx)), lang: langFromExt(absPath) };
}

function extractByPattern(absPath, startPattern, endPattern) {
  const lines = readSourceLines(absPath);
  const startRe = new RegExp(startPattern);
  const endRe = new RegExp(endPattern);

  const startMatches = [];
  for (let idx = 0; idx < lines.length; idx++) {
    if (startRe.test(lines[idx])) startMatches.push(idx);
  }

  if (startMatches.length === 0) {
    throw new SnippetError(`startPattern ${JSON.stringify(startPattern)} did not match any line in ${absPath}`);
  }
  if (startMatches.length > 1) {
    throw new SnippetError(
      `startPattern ${JSON.stringify(startPattern)} matched ${startMatches.length} lines in ${absPath} (must match exactly one)`
    );
  }

  const startIdx = startMatches[0];
  let endIdx = -1;
  for (let idx = startIdx; idx < lines.length; idx++) {
    if (endRe.test(lines[idx])) {
      endIdx = idx;
      break;
    }
  }

  if (endIdx === -1) {
    throw new SnippetError(
      `endPattern ${JSON.stringify(endPattern)} did not match any line at or after the startPattern match (line ${startIdx + 1}) in ${absPath}`
    );
  }

  return { lines: dedent(lines.slice(startIdx, endIdx + 1)), lang: langFromExt(absPath) };
}

function extractSnippet(recipeDir, marker) {
  const absPath = path.join(recipeDir, marker.snipPath);

  let extracted;
  if (marker.isFile) {
    extracted = extractByFile(absPath);
  } else if (marker.id) {
    extracted = extractById(absPath, marker.id);
  } else if (marker.config.startPattern && marker.config.endPattern) {
    extracted = extractByPattern(absPath, marker.config.startPattern, marker.config.endPattern);
  } else {
    throw new SnippetError(
      `SNIPSTART ${marker.snipPath} has no ":id" and no startPattern/endPattern — can't locate the snippet`
    );
  }

  let lines = extracted.lines;
  if (marker.config.selectedLines) {
    lines = applySelectedLines(lines, marker.config.selectedLines);
  }

  return { lines, lang: extracted.lang };
}

function renderFence(lines, lang) {
  return '```' + lang + '\n' + lines.join('\n') + '\n```';
}

function normalizeInner(text) {
  const trimmed = text.replace(/\r\n/g, '\n').replace(/[ \t]+\n/g, '\n');
  const lines = trimmed.split('\n');
  while (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const mode = argv[2];
  if (mode !== '--check' && mode !== '--fix') {
    console.error('Usage: sync-readme-snippets.js --check|--fix [path]');
    process.exit(1);
  }
  const targetPath = argv[3] || process.cwd();
  return { mode, targetPath };
}

function processReadme(readmePath, cookbookDir, mode) {
  const relativePath = path.relative(cookbookDir, readmePath);
  const recipeDir = path.dirname(readmePath);
  const content = fs.readFileSync(readmePath, 'utf8');

  let markers;
  try {
    markers = parseReadmeMarkers(content);
  } catch (e) {
    return { file: relativePath, errors: [e.message], fixed: [], skippedCount: 0 };
  }

  if (markers.length === 0) {
    return { file: relativePath, errors: [], fixed: [], skippedCount: 0, noMarkers: true };
  }

  const errors = [];
  const fixed = [];
  // Apply fixes back-to-front so earlier byte offsets stay valid as we splice.
  let newContent = content;
  const markersInReverse = [...markers].reverse();

  for (const marker of markersInReverse) {
    const label = marker.id ? `${marker.snipPath}:${marker.id}` : marker.snipPath;
    let expected;
    try {
      const { lines, lang } = extractSnippet(recipeDir, marker);
      expected = renderFence(lines, lang);
    } catch (e) {
      errors.push(`${label} (line ${marker.openLineNumber}): ${e.message}`);
      continue;
    }

    const expectedInner = expected + '\n';
    const inSync = normalizeInner(marker.currentInner) === normalizeInner(expectedInner);

    if (mode === '--check') {
      if (!inSync) {
        errors.push(`${label} (line ${marker.openLineNumber}): out of sync with source (run --fix)`);
      }
      continue;
    }

    // --fix
    if (!inSync) {
      newContent = newContent.slice(0, marker.openLineEnd) + expectedInner + newContent.slice(marker.closeLineStart);
      fixed.push(label);
    }
  }

  if (mode === '--fix' && fixed.length > 0) {
    fs.writeFileSync(readmePath, newContent, 'utf8');
  }

  return { file: relativePath, errors, fixed, skippedCount: 0, checkedCount: markers.length };
}

function main() {
  const { mode, targetPath } = parseArgs(process.argv);
  const cookbookDir = fs.statSync(targetPath).isDirectory() ? targetPath : path.dirname(targetPath);

  console.log(`[sync-readme-snippets] Mode: ${mode.slice(2)} — scanning: ${cookbookDir}\n`);

  if (!fs.existsSync(cookbookDir)) {
    console.error(`Error: path not found: ${cookbookDir}`);
    process.exit(1);
  }

  const readmes = findTopLevelReadmes(cookbookDir);
  const failures = [];
  let totalFixed = 0;
  let totalChecked = 0;

  for (const readmePath of readmes) {
    const result = processReadme(readmePath, cookbookDir, mode);

    if (result.noMarkers) {
      continue; // quiet — most recipe READMEs have no snippet markers yet
    }

    if (result.errors.length > 0) {
      failures.push(result);
      console.error(`❌ ${result.file}`);
      for (const err of result.errors) console.error(`    → ${err}`);
    } else if (result.fixed && result.fixed.length > 0) {
      console.log(`🔧 ${result.file} — fixed: ${result.fixed.join(', ')}`);
    } else {
      console.log(`✓ ${result.file} (${result.checkedCount} snippet(s) in sync)`);
    }

    totalFixed += result.fixed ? result.fixed.length : 0;
    totalChecked += result.checkedCount || 0;
  }

  console.log('');

  if (mode === '--check') {
    if (failures.length > 0) {
      const errorCount = failures.reduce((n, f) => n + f.errors.length, 0);
      console.error(`❌ Found ${errorCount} snippet drift error(s) across ${failures.length} file(s)`);
      process.exit(1);
    }
    console.log(`✅ All ${totalChecked} snippet(s) are in sync with source`);
    process.exit(0);
  }

  // --fix
  if (failures.length > 0) {
    const errorCount = failures.reduce((n, f) => n + f.errors.length, 0);
    console.error(`❌ ${errorCount} snippet(s) could not be fixed automatically (see above)`);
    process.exit(1);
  }
  console.log(`✅ Fixed ${totalFixed} snippet(s); ${totalChecked - totalFixed} were already in sync`);
  process.exit(0);
}

main();
