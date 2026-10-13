#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build.py
cd webapp
exec python3 -m http.server "${PORT:-8645}" --bind 127.0.0.1
