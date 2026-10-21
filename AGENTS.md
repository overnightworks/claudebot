# AGENTS.md

## Quality gate

CI runs the shared `overnightworks/quality` gate (`quality` job in
`.github/workflows/ci.yml`, pinned by commit SHA). Its `quality / verdict`
check is the required check on `main`; it follows the mandatory jobs named in
`quality/config.toml` and the single SonarCloud scan the gate owns.
