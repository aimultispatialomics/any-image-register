# Contributing

Thanks for helping improve `any-image-register`! This guide covers the
contribution workflow and the review checklist we apply to every PR.

## Workflow

1. Fork the repository and branch from `main`.
2. Keep each PR focused on one change: a feature module, a performance
   improvement, a bug fix, or a docs update.
3. Commit messages follow Conventional Commits (`feat:`, `perf:`,
   `fix:`, `docs:`, `test:`, `chore:`).
4. Open the PR with a short description of *why* the change is needed,
   not only *what* changed.

## Review checklist

Reviewers look at the following before merging — feel free to discuss
any of these points in the PR thread:

- **Correctness** — does the transform convention (forward vs inverse
  mapping) match the rest of the package?
- **Dependencies** — the core package must stay numpy-only; optional
  accelerators (scipy, cupy) go behind guarded imports.
- **Performance** — changes to hot paths (pyramid construction, metric
  evaluation, warping) should include a before/after timing note.
- **Tests** — new modules need unit tests under `tests/`.

## Running tests

```bash
python -m pytest tests/
```
