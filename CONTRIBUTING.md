# Contributing to SentinelCam

Thanks for your interest in the project. This document covers how to get a working development
environment, what the code conventions are, and what a pull request needs to pass.

## Getting set up

The fastest path is Docker Compose — see [Quick start](README.md#quick-start) in the README. For
day-to-day development on the backend or frontend you will usually want the native setup described
under [Local development](README.md#local-development-without-docker), because it gives you hot
reload and direct access to the test databases.

You will need:

- Python 3.11+
- Node 20+
- PostgreSQL 16 (two databases: `sentinelcam` and `sentinelcam_test`)
- Docker, if you want to run the end-to-end suite

## Project layout

| Directory | What lives there |
|---|---|
| `backend/` | FastAPI service — routes, SQLAlchemy models, detection/recording/notification services, Alembic migrations, pytest suite |
| `frontend/` | React + Vite SPA — pages, components, Zustand stores, Vitest suite |
| `ml/` | Standalone training pipeline. Not imported by the backend; it produces model artifacts |
| `e2e/` | Playwright suite that drives the real Docker Compose stack |

`ml/` is deliberately decoupled: the backend consumes an exported artifact via a config path and
knows nothing about how it was produced. Please keep it that way — no imports from `ml/` into
`backend/`.

## Before you open a pull request

Run the checks that CI runs. All of them must pass.

```bash
# Backend
cd backend && source venv/bin/activate
ruff check .
pytest

# Frontend
cd frontend
npm run lint
npm run test
npm run build

# End-to-end (requires Docker)
cd e2e
npm test
```

If you touched a SQLAlchemy model, generate a migration and commit it with the change:

```bash
cd backend
alembic revision --autogenerate -m "describe the change"
```

Review the generated migration by hand — autogenerate does not detect every change, and any
migration that alters existing columns must preserve existing data (see
`alembic/versions/dc14a9d0..._add_rbac_roles...py` for a worked example that backfills before
dropping a column).

## Coding conventions

**Python**

- Formatted and linted with `ruff`; the configuration in `backend/pyproject.toml` is the source of
  truth. Don't add a competing formatter.
- Type-annotate function signatures. Pydantic schemas define the API contract — routes should
  return schema objects, not ad-hoc dicts.
- Configuration goes through `app/config.py` (pydantic-settings). Never read `os.environ` directly
  in a route or service, and never hardcode a credential, host or path.
- Authorization is enforced with the `require_admin` / `require_operator` dependencies on the
  route. Do not rely on the frontend to hide a capability.

**JavaScript / React**

- Linted with ESLint (`frontend/eslint.config.js`). Function components and hooks only.
- Server state goes through the Axios client in `src/api/client.js` so auth interception and error
  handling stay in one place. Global UI state goes in a Zustand store under `src/store/`.
- Tailwind utility classes for styling; no separate CSS modules.

**Tests**

- Backend tests run against a real PostgreSQL database, not sqlite or a mock. Add fixtures to
  `backend/tests/conftest.py` rather than constructing state inline in each test.
- Frontend tests use Vitest with Testing Library — query by accessible role and text, not by test
  ids or class names, unless there is genuinely no accessible handle.
- Every bug fix should come with a test that fails without the fix.
- Don't write a test that asserts something the code cannot actually do. If a scenario needs
  hardware or a real model detection that CI cannot provide, say so in the test file rather than
  faking the assertion (see `e2e/README.md` for the existing manual/automated split).

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add in-video timestamp persistence for upload analysis
fix: prevent stale upload state after a failed analysis
test: expand e2e coverage of the results view
docs: document the fall detector training workflow
chore: pin ultralytics to a known-good version
refactor: extract the clip writer from the recording engine
perf: stride frames during upload analysis
```

Keep the subject line under ~72 characters and in the imperative mood. Explain *why* in the body
when the reason isn't obvious from the diff.

## Pull requests

- One logical change per PR. A refactor and a feature in the same diff is two PRs.
- Describe what changed, why, and how you verified it.
- Update the README when you change setup steps, environment variables or the test counts.
- If your change adds a limitation or a known-issue, add it to the README's
  [Known limitations](README.md#known-limitations) rather than leaving it undocumented.

## What not to commit

- Any real `.env` file, credential, API key or token. The repository has never contained one and
  should stay that way — see [SECURITY.md](SECURITY.md).
- Datasets, training runs, checkpoints, or logs under `ml/data/`, `ml/runs/` or `ml/logs/`. These
  are gitignored; the download and preparation scripts are the reproducible path.
- Model weights, other than the small exported artifacts already in `ml/exported/`.
- Screenshots or recordings containing real people or identifiable locations.

## Reporting bugs and requesting features

Open a GitHub issue with reproduction steps, the expected and actual behaviour, and relevant log
output. For anything security-sensitive, follow [SECURITY.md](SECURITY.md) instead — please don't
file a public issue.
