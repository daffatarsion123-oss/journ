#!/usr/bin/env bash
# Print the detected compute backend on container start, then exec the command.
# This makes the ROCm-vs-CUDA-vs-CPU situation obvious immediately and matches
# the codebase's runtime backend-detection design.
set -e

echo "============================================================"
echo " eegrag container | $(python --version 2>&1)"
python - <<'PY' || echo "  (backend probe skipped: $?)"
try:
    from eegrag.utils.backend import detect_backend
    print("  backend:", detect_backend().summary())
except Exception as e:
    print("  backend probe failed:", repr(e))
PY
echo "============================================================"

# If no explicit command was given, fall through to the image CMD.
if [ "$#" -eq 0 ]; then
    exec bash
fi
exec "$@"
