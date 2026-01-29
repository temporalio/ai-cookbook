#!/usr/bin/env node
'use strict';

/**
 * Validates front matter in AI cookbook recipe README files.
 *
 * Top-level recipe READMEs (e.g., agents/recipe-name/README.md) must have valid front matter.
 * Nested READMEs within a recipe (e.g., agents/recipe-name/tools/README.md) are exempt.
 */

const fs = require('fs/promises');
const path = require('path');

// YAML parsing with built-in support - using a simple parser to avoid external dependencies
function parseSimpleYaml(text) {
  const lines = text.split('\n');
  const result = {};

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trimEnd();

    // Skip empty lines and comments
    if (!line || line.startsWith('#')) continue;

    // Match key-value pairs
    const match = line.match(/^([\w-]+):\s*(.*)$/);
    if (!match) continue;

    const [, key, value] = match;

    // Handle arrays (lines starting with -)
    if (value === '') {
      const arrayValues = [];
      i++;
      while (i < lines.length) {
        const arrayLine = lines[i].trim();
        if (!arrayLine.startsWith('-')) {
          i--;
          break;
        }
        arrayValues.push(arrayLine.slice(1).trim());
        i++;
      }
      result[key] = arrayValues;
    } else if (value.startsWith('[') && value.endsWith(']')) {
      // Inline array
      result[key] = value.slice(1, -1).split(',').map(v => v.trim());
    } else {
      // Simple value
      result[key] = value;
    }
  }

  return result;
}

// Get the cookbook directory (current directory or specified path)
const COOKBOOK_DIR = process.argv[2] || process.cwd();

/**
 * Extract and validate front matter from a README file
 * @param {string} content - The file content
 * @returns {{ valid: boolean, error?: string, data?: object }}
 */
function validateFrontMatter(content) {
  // Check for HTML comment front matter (<!-- ... -->)
  const commentPattern = /^\s*<!--([\s\S]*?)-->/;
  const match = commentPattern.exec(content);

  if (!match) {
    return { valid: false, error: 'missing front matter comment' };
  }

  const commentBody = match[1].replace(/\r/g, '').trim();

  // Try to parse as YAML
  let data;
  try {
    // Normalize multi-line values (same logic as sync script)
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
    const normalizedComment = normalizedLines
      .join('\n')
      .replace(/(^|\n)([ \t]*[\w-]+):(?=\S)/g, (full, prefix, key) => `${prefix}${key}: `);

    data = parseSimpleYaml(normalizedComment);
  } catch (error) {
    return { valid: false, error: `invalid YAML in front matter: ${error.message}` };
  }

  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    return { valid: false, error: 'front matter must be a YAML object' };
  }

  // Check for required fields
  const requiredFields = ['description'];
  const missingFields = requiredFields.filter(field => !data[field]);

  if (missingFields.length > 0) {
    return { valid: false, error: `missing required field(s): ${missingFields.join(', ')}` };
  }

  return { valid: true, data };
}

/**
 * Check if a README has a parent directory that also contains a README.
 * If true, this README is nested and doesn't need front matter validation.
 * @param {string} readmePath - Absolute path to README.md
 * @param {string} cookbookRoot - Absolute path to cookbook root
 * @returns {Promise<boolean>}
 */
async function hasParentReadme(readmePath, cookbookRoot) {
  // Start from the directory containing this README
  let currentDir = path.dirname(readmePath);

  // Normalize paths for comparison
  const normalizedRoot = path.resolve(cookbookRoot);

  // Walk up the directory tree
  while (currentDir !== normalizedRoot && currentDir !== path.dirname(currentDir)) {
    // Move to parent directory
    currentDir = path.dirname(currentDir);

    // Stop if we've reached the cookbook root (don't check the root itself)
    if (currentDir === normalizedRoot) {
      break;
    }

    // Stop if we've somehow passed the cookbook root
    if (currentDir.length < normalizedRoot.length) {
      break;
    }

    // Check if parent directory has a README.md
    const parentReadme = path.join(currentDir, 'README.md');
    try {
      await fs.access(parentReadme);
      // Found a parent README - this README is nested
      return true;
    } catch {
      // No README in this parent directory, continue checking
    }
  }

  // No parent README found - this is a top-level README
  return false;
}

/**
 * Find all README.md files in the cookbook
 * @param {string} rootDir - Root directory to search
 * @returns {Promise<string[]>}
 */
async function findReadmeFiles(rootDir) {
  const stack = [rootDir];
  const readmes = [];

  while (stack.length > 0) {
    const current = stack.pop();
    let entries;

    try {
      entries = await fs.readdir(current, { withFileTypes: true });
    } catch (error) {
      // Skip directories we can't read
      continue;
    }

    for (const entry of entries) {
      if (entry.name === '.git' || entry.name === '.github') {
        continue;
      }

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

/**
 * Main validation function
 */
async function main() {
  console.log(`[validate-frontmatter] Checking cookbook at: ${COOKBOOK_DIR}\n`);

  // Check if cookbook directory exists
  try {
    await fs.access(COOKBOOK_DIR);
  } catch (error) {
    console.error(`Error: Cookbook directory not found: ${COOKBOOK_DIR}`);
    process.exit(1);
  }

  // Find all README files
  const allReadmes = await findReadmeFiles(COOKBOOK_DIR);

  if (allReadmes.length === 0) {
    console.warn('Warning: No README.md files found in cookbook');
    process.exit(0);
  }

  // Filter to top-level recipe READMEs (those without a parent README)
  const recipeReadmes = [];
  for (const readme of allReadmes) {
    // Skip root-level README (repository documentation)
    if (path.dirname(readme) === path.resolve(COOKBOOK_DIR)) {
      continue;
    }

    const isNested = await hasParentReadme(readme, COOKBOOK_DIR);
    if (!isNested) {
      recipeReadmes.push(readme);
    }
  }

  console.log(`Found ${recipeReadmes.length} top-level recipe README(s) to validate`);
  console.log(`(${allReadmes.length - recipeReadmes.length} nested README(s) skipped)\n`);

  // Validate each recipe README
  const errors = [];

  for (const readmePath of recipeReadmes) {
    const relativePath = path.relative(COOKBOOK_DIR, readmePath);

    let content;
    try {
      content = await fs.readFile(readmePath, 'utf8');
    } catch (error) {
      errors.push({ file: relativePath, error: `cannot read file: ${error.message}` });
      continue;
    }

    const validation = validateFrontMatter(content);

    if (!validation.valid) {
      errors.push({ file: relativePath, error: validation.error });
    } else {
      console.log(`✓ ${relativePath}`);
    }
  }

  // Report results
  console.log('');

  if (errors.length > 0) {
    console.error(`❌ Found ${errors.length} error(s):\n`);
    for (const { file, error } of errors) {
      console.error(`  ${file}`);
      console.error(`    → ${error}\n`);
    }
    process.exit(1);
  } else {
    console.log(`✅ All ${recipeReadmes.length} recipe README(s) have valid front matter`);
    process.exit(0);
  }
}

// Run the validation
main().catch(error => {
  console.error('Unexpected error:', error);
  process.exit(1);
});
