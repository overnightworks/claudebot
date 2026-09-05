#!/usr/bin/env bash
set -euo pipefail

if ! systemctl --user is-enabled --quiet agent-telegram.service 2>/dev/null; then
    echo "Bot service is not installed."
    exit 0
fi

systemctl --user disable --now agent-telegram.service >/dev/null
echo "Bot service stopped and disabled."
