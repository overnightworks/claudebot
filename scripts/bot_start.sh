#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
UNIT="$REPOSITORY_ROOT/deploy/agent-telegram.service"

cd "$REPOSITORY_ROOT"

umask 077
[[ -x "$PYTHON" ]] || { echo "Missing bot interpreter: $PYTHON" >&2; exit 1; }

for config_file in "$HOME/.secrets/claudebot.env" "$HOME/.secrets/claudebot-runtime.env"; do
    [[ -f "$config_file" ]] || { echo "Missing configuration: $config_file" >&2; exit 1; }
    mode="$(stat -c '%a' "$config_file")"
    [[ "$mode" == "600" ]] || {
        echo "Refusing configuration with mode $mode: $config_file" >&2
        exit 1
    }
done

# The project is not installed into the virtualenv (pyproject: package =
# false), so the tests reach the package through the path.
PYTHONPATH="$REPOSITORY_ROOT/src" "$PYTHON" -m unittest discover -s "$REPOSITORY_ROOT/tests" -q
"$PYTHON" -m py_compile "$REPOSITORY_ROOT/src/claudebot/agent_bridge.py" \
    "$REPOSITORY_ROOT/src/claudebot/telegram_bot.py"

systemctl --user link --force "$UNIT" >/dev/null
systemctl --user daemon-reload
systemctl --user enable agent-telegram.service >/dev/null
systemctl --user restart agent-telegram.service

for _ in {1..40}; do
    if systemctl --user is-active --quiet agent-telegram.service; then
        echo "Bot service is active."
        exit 0
    fi
    sleep 0.25
done

echo "Bot service did not become active." >&2
systemctl --user status agent-telegram.service --no-pager >&2 || true
exit 1
