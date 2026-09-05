#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
UNIT="$SCRIPT_DIR/agent-telegram.service"
REMOTE_UNIT="$SCRIPT_DIR/claude-remote-control.service"

cd "$SCRIPT_DIR"

umask 077
[[ -x "$PYTHON" ]] || { echo "Missing bot interpreter: $PYTHON" >&2; exit 1; }

for config_file in "$SCRIPT_DIR/.env" "$SCRIPT_DIR/runtime.env"; do
    [[ -f "$config_file" ]] || { echo "Missing configuration: $config_file" >&2; exit 1; }
    mode="$(stat -c '%a' "$config_file")"
    [[ "$mode" == "600" ]] || {
        echo "Refusing configuration with mode $mode: $config_file" >&2
        exit 1
    }
done

"$PYTHON" -m unittest discover -s "$SCRIPT_DIR/tests" -q
"$PYTHON" -m py_compile "$SCRIPT_DIR/agent_bridge.py" "$SCRIPT_DIR/telegram_bot.py"

systemctl --user link --force "$UNIT" >/dev/null
systemctl --user link --force "$REMOTE_UNIT" >/dev/null
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
