<!-- Copilot instructions for AI coding agents working in this repository -->
# Repository Overview

- This repository currently contains a single placeholder file named `Data Cleaning` at the project root. There are no code, notebooks, or dependency manifests yet. Before making broad changes, ask the human owner which workflow they expect (notebooks-first, package-first, or pipeline-first).

# Goals for AI agents

- Be conservative by default: propose small, incremental changes and request approval for structural or destructive edits (especially to data).
- Prefer creating new files under clear folders rather than modifying many files at once. Suggested layout to propose:
  - `notebooks/` — exploratory analysis and iteration
  - `src/` — reusable Python modules and scripts
  - `data/` — raw and processed data (do not commit large datasets without consent)
  - `tests/` — unit/integration tests
  - `requirements.txt` or `pyproject.toml` — dependency pins

# Project-specific conventions and constraints

- There are no existing conventions encoded yet; when proposing conventions, show one concrete example and apply it consistently. For example, when adding a notebook, include a short `README.md` describing its purpose and required data files.
- Windows path awareness: authoring should use pathlib and OS-agnostic paths. When writing examples, use forward-slash repo-relative paths (the maintainer works on Windows but CI may be cross-platform).

# How to work with this repo (step-by-step)

1. Ask the user whether they want a notebook-first or package-first structure.
2. Propose a minimal scaffold (one notebook, one `src/` module, `requirements.txt`, and `.gitignore`) and wait for approval.
3. If approved, add files in a single focused commit with a clear commit message and short PR description.

# Coding style and PR behavior

- Keep changes small and well-scoped (one logical change per PR). Use descriptive commit messages like: `chore(scaffold): add notebooks, src, requirements`.
- When adding dependencies, list why each is needed and pin versions in `requirements.txt`.

# Data handling rules

- Never create or commit large data files without explicit permission. If processing/data examples are needed, use small synthetic samples under `data/sample/`.
- Document data sources and any transformations in a `notebooks/README.md` or `DATA.md`.

# Testing, running, and debugging

- There are no test or build scripts yet. If you add tests, also add a short `README.md` describing how to run them (example: `python -m pytest tests/`).
- Prefer adding small runnable examples (a tiny script under `examples/`) that demonstrate the change.

# Communication and clarification

- When uncertain about project goals, ask one or two targeted questions (e.g., "Should I scaffold the repo with Jupyter notebooks or build an importable package?").
- Summarize proposed changes in the PR description and call out any assumptions.

# Files to reference when following examples

- `Data Cleaning` — current placeholder file at the repo root. Mention this file when proposing scaffolds or reorganizations.

# If you update this file

- Merge any existing owner-provided guidance instead of replacing it wholesale. When in doubt, preserve human-written content and append clarifying examples.

— End of instructions —
