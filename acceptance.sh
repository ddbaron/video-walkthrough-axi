#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
video-walkthrough-axi --version
video-walkthrough-axi --help
python -m pytest "$root/tests" "$@"
