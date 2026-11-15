#!/usr/bin/env bash
# Set Kip up on a Raspberry Pi 4 (Raspberry Pi OS 64-bit, Trixie), or update an existing setup.
# Run from anywhere as your normal user: bash deploy/install.sh   (it asks for sudo when it needs it)
# Safe to run again after a git pull: it keeps backend/.env and backend/memory/kip.db.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${OLLAMA_MODEL:-qwen3.5:2b-q4_K_M}"
say() { printf '\n== %s\n' "$*"; }
fail() { printf '\nStopped: %s\n' "$*" >&2; exit 1; }

say "Checking the Pi"
[ "$(id -u)" -ne 0 ] || fail "run this as your normal user, not root (it uses sudo where needed)"
[ "$(uname -m)" = "aarch64" ] || echo "Warning: expected 64-bit Raspberry Pi OS (aarch64), found $(uname -m)."
python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
  || fail "Kip needs Python 3.12 or newer; Raspberry Pi OS 13 (Trixie) has 3.13."

missing=()
python3 -m venv --help >/dev/null 2>&1 || missing+=(python3-venv)
command -v node >/dev/null 2>&1 || missing+=(nodejs npm)
if [ "${#missing[@]}" -gt 0 ]; then
  say "Installing ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi

say "Checking Ollama and the model ($MODEL)"
if ! command -v ollama >/dev/null 2>&1; then
  fail "Ollama is not installed. Install it with its official script, then run this again:
  curl -fsSL https://ollama.com/install.sh | sh"
fi
systemctl is-active --quiet ollama || sudo systemctl enable --now ollama
ollama pull "$MODEL"

say "Backend: Python environment and packages"
cd "$REPO/backend"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created backend/.env from the example: add your keys there (GROQ_API_KEY and the feed keys)."
fi
chmod 600 .env  # it holds API keys
mkdir -p memory

say "Frontend: building the page"
cd "$REPO/frontend"
if node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)||(a===20&&b>=19)?0:1)'; then
  npx --yes pnpm@10 install --frozen-lockfile
  npx --yes pnpm@10 build
elif [ -f dist/index.html ]; then
  echo "Node $(node --version) is too old to build (needs 20.19+ or 22.12+); keeping the existing frontend/dist."
else
  fail "Node $(node --version) is too old to build the page (needs 20.19+ or 22.12+).
  Build it on your computer (cd frontend && pnpm build) and copy frontend/dist to the Pi, then run this again."
fi

say "Checking the model answers in the shape Kip needs"
cd "$REPO/backend"
.venv/bin/python -m app.check_model || echo "Warning: the model check failed; Kip will still start. See the README's troubleshooting."

say "Installing the kip service (starts on boot)"
sed -e "s|@USER@|$(id -un)|g" -e "s|@REPO@|$REPO|g" "$REPO/deploy/kip.service" | sudo tee /etc/systemd/system/kip.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable kip >/dev/null
sudo systemctl restart kip

for _ in $(seq 1 60); do  # the first start loads the router and the embedding model
  curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break
  sleep 2
done
if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  say "Kip is running"
  cat <<EOF
On the Pi:              open http://127.0.0.1:8000
From another computer:  ssh -L 8000:127.0.0.1:8000 $(id -un)@$(hostname).local
                        then open http://localhost:8000 there
Logs:                   journalctl -u kip -f
Restart / stop:         sudo systemctl restart kip / sudo systemctl stop kip
EOF
else
  fail "Kip did not answer on port 8000 yet. See: journalctl -u kip -n 50"
fi
