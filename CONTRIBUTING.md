# Contributing

## Setup

1. Create a Python 3.13 virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run smoke tests:

   ```bash
   python -m pytest tests/test_smoke.py -q
   ```

## Development Guidelines

- Follow snake_case naming for files and functions.
- Add module-level and public API docstrings (Google style).
- Add type hints for public functions.
- Avoid hardcoded absolute paths; use `pathlib.Path`.
- Keep generated artifacts out of git.

## Pull Requests

- Keep changes focused and atomic.
- Update `README.md` when behavior or CLI changes.
- Include a short verification summary in the PR description.
