#!/usr/bin/env bash
set -euo pipefail

systemctl --user status agent-telegram.service --no-pager || true
journalctl --user -u agent-telegram.service -n 20 --no-pager \
    | sed -E 's#https://api\.telegram\.org/bot[^/]+/#https://api.telegram.org/bot<redacted>/#g'
