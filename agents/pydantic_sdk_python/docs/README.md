# Sample Documentation

Place markdown files here for the bot to search.

## Quick Setup

Download a few sample markdown files from any docs:

```bash
# Download directly from GitHub
curl -o docs/workflows.md https://raw.githubusercontent.com/temporalio/documentation/main/docs/develop/python/core-application.md

curl -o docs/activities.md https://raw.githubusercontent.com/temporalio/documentation/main/docs/develop/python/failure-detection.md

curl -o docs/testing.md https://raw.githubusercontent.com/temporalio/documentation/main/docs/develop/python/testing-suite.md
```

Or manually download any markdown files and place them in this directory.

The bot will read all `*.md` files from this folder.
