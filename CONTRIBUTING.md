# Contributing to TPU agent suite

Thank you for your interest in the TPU agent suite repository!

## Our Contribution Policy

At this time, we are not accepting external pull requests or code contributions.

## How You Can Help

Your feedback is invaluable for improving the quality of these agents and
skills. Please feel free to open an issue on GitHub to report bugs, suggest
features, or request improvements.

## Setting up linting, formatting, and static type checking

```bash
pip install pre-commit pyyaml

# Install hooks (the config declares pre-commit, pre-push, and commit-msg hook types)
pre-commit install

# Manually run pre-commit (you shouldn't need to do this as it's run
# automatically on every commit)
pre-commit run --all-files
```

The full development setup, `make` targets, and CI gates are described in
[`docs/developing.md`](docs/developing.md).
