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
# false), so the tests reach the package and the gate through the path.
PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT/scripts" "$PYTHON" -m unittest discover -s "$REPOSITORY_ROOT/tests" -q
"$PYTHON" -m py_compile "$REPOSITORY_ROOT/src/claudebot/agent_bridge.py" \
    "$REPOSITORY_ROOT/src/claudebot/telegram_bot.py"

# Both links are forced: after the unit file moves, the stale symlinks in
# ~/.config/systemd/user and its default.target.wants still point at the old
# path, and an unforced link or enable aborts on the conflict -- under `set -e`
# before the restart, leaving the live bot on a unit file that no longer exists.
systemctl --user link --force "$UNIT" >/dev/null
systemctl --user enable --force agent-telegram.service >/dev/null
systemctl --user daemon-reload
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
