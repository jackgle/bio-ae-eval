#!/usr/bin/env bash
set -Eeuo pipefail

# config (override with env vars if you want)
PY_BOOT_ENV="${PY_BOOT_ENV:-py311}"                     # conda env with python 3.11
ENV_DIR="${ENV_DIR:-$PWD/venv}"                         # venv target
REPO_URL="${REPO_URL:-https://github.com/bioacoustic-ai/bacpipe.git}"
REPO_DIR="${REPO_DIR:-$PWD/bacpipe}"
KERNEL_NAME="${KERNEL_NAME:-bioacoustic-embedding-eval}"
KERNEL_LABEL="${KERNEL_LABEL:-Python ($KERNEL_NAME)}"

say() { printf '\n>>> %s\n' "$*"; }

say "ensure bootstrap env exists (python 3.11)"
conda run -n "$PY_BOOT_ENV" python -V >/dev/null 2>&1 || conda create -y -n "$PY_BOOT_ENV" python=3.11

say "locate py311 interpreter"
PY311=$(conda run -n "$PY_BOOT_ENV" python -c "import sys,shlex; print(shlex.quote(sys.executable))")

say "create venv (idempotent)"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  conda run -n "$PY_BOOT_ENV" "$PY311" -m venv "$ENV_DIR"
fi

VENV_PY="$ENV_DIR/bin/python"

say "seed build tools + numpy in venv"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install "numpy<2"

# regex filter for lines to skip in requirements (leading spaces allowed)
FILTER_RE='^[[:space:]]*(pyqt6(-qt6|-sip)?|pyside6|qtpy|pyqtgraph|usearch)\b'

say "install root project deps (filtered) into venv"
ROOT_REQ="$PWD/requirements.txt"
if [ -f "$ROOT_REQ" ]; then
  ROOT_REQ_F="$(mktemp)"
  grep -viE "$FILTER_RE" "$ROOT_REQ" > "$ROOT_REQ_F"
  "$VENV_PY" -m pip install -r "$ROOT_REQ_F"
fi

say "clone or update bacpipe repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" pull --ff-only || true
fi
cd "$REPO_DIR"

say "install bacpipe deps (filtered) into venv"
REQS_FILTERED="$(mktemp)"
grep -viE "$FILTER_RE" requirements.txt > "$REQS_FILTERED"
"$VENV_PY" -m pip install -r "$REQS_FILTERED"

say "install bacpipe (no deps to avoid re-pulling usearch)"
"$VENV_PY" -m pip install -e . --no-deps

say "install kernel + pytest"
"$VENV_PY" -m pip install ipykernel pytest
"$VENV_PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL"

say "quick import check"
"$VENV_PY" - <<'PY'
import sys
try:
    import bacpipe
    print("import bacpipe: ok")
except Exception as e:
    print("import bacpipe: failed:", e, file=sys.stderr)
    sys.exit(1)
PY

echo
echo "done. kernel installed: $KERNEL_LABEL"
echo "to test like your manual run:"
echo "  cd \"$REPO_DIR\""
echo "  pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py"
