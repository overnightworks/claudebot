# claudebot

Private Telegram bridge to local Claude Code and Codex CLI sessions.
One bot, one systemd user service: `agent-telegram.service`.

## Run

```
./bot_start.sh    # stdlib unittest preflight, compile check, link + enable + restart the service
./bot_status.sh   # service status and the last 20 journal lines (token redacted)
./bot_stop.sh     # disable and stop
```

Telegram allows exactly one polling instance per bot token. A second
service on the same token produces `Conflict: terminated by other getUpdates
request` in both journals and neither bot answers reliably.

## Configuration

Secrets and runtime settings live outside the repo in `~/.secrets`, mode 600:

| File | Keys |
|---|---|
| `~/.secrets/claudebot.env` | `TELEGRAM_BOT_TOKEN`, `CLAUDE_BIN` |
| `~/.secrets/claudebot-runtime.env` | `TELEGRAM_ALLOWED_USER_IDS`, `AGENT_PROJECT_DIR`, `AGENT_PROVIDER`, `CODEX_MODEL`, `AGENT_TIMEOUT` |

`agent-telegram.service` loads both via `EnvironmentFile`.

## Bot commands

`/start`, `/provider`, `/takeover`, `/new`, `/clear`, `/sessions`,
`/session`, `/status`, `/stop`, `/claude_remote`, `/claude_remote_stop`,
`/player`. Any other message is forwarded to the active agent session.

## Install

```
uv sync --locked --no-build --group dev
```

Creates `.venv`, the same `.venv/bin/python` the systemd unit and
`bot_start.sh` already use.

## Tests

```
.venv/bin/python -m pytest
```

CI runs the same command with branch coverage and keeps the resulting
`coverage.xml` as an artifact. Once `SONAR_TOKEN` is configured, the sonar
job arms itself: it uploads that coverage and a "SonarCloud open findings"
step fails the build on any open finding in scope. Until then every step of
that job is skipped — the job says so with a notice — and SonarCloud's
Automatic Analysis measures the repository while it is private.
