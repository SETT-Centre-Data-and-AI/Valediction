# Usage in Repos

Use this template to set up a downstream repository against the stable workflows release.

Template location:

- `repo_template/.github/workflows`

## Quick Setup

1. Copy `repo_template/.github/workflows` to `/.github/workflows` in your downstream repository.
2. Confirm these files exist:

- `/.github/workflows/config.yaml`
- `/.github/workflows/orchestrator.yaml`
- `/.github/workflows/sync-from-public.yaml`
- `/.github/workflows/pre-install.sh`
- `/.github/workflows/README.md`

## Configure `config.yaml`

Set configuration values once in this file:

- `package-name`
- `package-slug`
- `private-repo`
- `public-repo`
- `build-smoke-python-version`
- `version-check-python-version`
- `publish-python-version`
- `publish-on-release`
- `sync-to-public`
- `test-matrix-json` (optional compact JSON; leave empty for central defaults)

Both `orchestrator.yaml` and `sync-from-public.yaml` read from this file. Edit placeholder values to match your setup.

## `orchestrator.yaml`

No identity or runtime settings are configured in this file; it reads all values from `config.yaml`.

This file calls the central stable orchestrator:

- `SETT-Centre-Data-and-AI/workflows/.github/workflows/central-orchestrator.yaml@release`

## `sync-from-public.yaml`

Use this workflow for manual pullback from a selected public branch to private.

## Optional `pre-install.sh`

Use this script if your tests need extra setup before package install.

## Required Secrets

Set repository or organisation secrets as needed:

- `MANAGEMENT_TOKEN` for cross-repository sync operations
- `PYPI_TOKEN` for publishing from public repositories

No extra repository variables are required for default use.
