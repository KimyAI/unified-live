# Activate the prepared CI

`tests.yml` contains the Python 3.11 Windows/Linux test matrix, lint, full tests
and real Qt offscreen smoke run. It is intentionally outside `.github/workflows`:
GitHub rejected the initial push because the connected OAuth app lacks the
`workflow` scope. All application code is published; the CI is currently inactive.

From an authenticated terminal with permission to add workflows:

```powershell
gh auth refresh -h github.com -s workflow
New-Item -ItemType Directory -Force .github/workflows
Copy-Item docs/ci/tests.yml .github/workflows/tests.yml
git add .github/workflows/tests.yml
git commit -m "ci: activate Windows and Linux validation"
git push
```

Authentication is interactive and belongs to the repository owner. Never paste
tokens into source or configuration. Once a run exists, inspect its actual result
before marking Windows validation complete. GitHub-hosted Windows CI still lacks
the user's RTX 4090, authorized AI models and physical virtual-device routing.
